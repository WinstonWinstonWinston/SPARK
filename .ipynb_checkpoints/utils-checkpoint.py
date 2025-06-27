import parmed as pmd
import torch
import numpy as np
from system.topology import Topology
from forces.twobody import MixtureLJ, Harmonic_Bond, MixtureCoulomb
from forces.threebody import Harmonic_Angle
from forces.fourbody import Dihedral
import system.units as units
import system.box as box
import system.system as sys
import math

def save_xyz(
    trajectory: torch.Tensor,
    atomic_numbers: list[int] | torch.Tensor,
    prefix: str = "output",
):
    """
    Save a trajectory of shape (steps, B, N, 3) as one XYZ file per batch,
    using atomic numbers for proper element symbols.

    Parameters
    ----------
    trajectory : torch.Tensor
        Tensor of shape (steps, B, N, 3)
    atomic_numbers : list[int] or torch.Tensor
        Atomic numbers of shape (N,)
    prefix : str
        Output file prefix; files will be named '{prefix}_{b}.xyz'
    """
    steps, B, N, _ = trajectory.shape

    if isinstance(atomic_numbers, torch.Tensor):
        atomic_numbers = atomic_numbers.tolist()

    # Periodic table mapping for atomic numbers 1–20, fallback to "X"
    periodic_table = { 0: "H",
        1: "H",  2: "He", 3: "Li", 4: "Be", 5: "B",  6: "C",  7: "N",  8: "O",  9: "F", 10: "Ne",
        11: "Na",12: "Mg",13: "Al",14: "Si",15: "P",16: "S",17: "Cl",18: "Ar",19: "K", 20: "Ca",
    }

    symbols = [periodic_table.get(z, "X") for z in atomic_numbers]

    for b in range(B):
        with open(f"{prefix}_{b}.xyz", "w") as f:
            for step in range(steps):
                f.write(f"{N}\n")
                f.write(f"Frame {step}\n")
                for atom in range(N):
                    x, y, z = trajectory[step, b, atom]
                    symbol = symbols[atom]
                    f.write(f"{symbol} {x:.3f} {y:.3f} {z:.3f}\n")

def save_pdb_with_bonds(pos: torch.Tensor, atomic_numbers: list[int], top, filename="output.pdb"):
    """
    Save a single snapshot as a PDB file with CONECT bonds.
    
    Parameters
    ----------
    pos : torch.Tensor
        (N, 3) atom positions
    atomic_numbers : list[int]
        Atomic numbers for each atom
    top : Topology
        Object with get_arity(2) -> dict[label -> set[(i,j)]]
    filename : str
        Output filename
    """
    periodic_table = {
        1: "H", 6: "C", 7: "N", 8: "O", 16: "S", 17: "Cl", 14: "Si", 15: "P"
    }
    symbols = [periodic_table.get(z, "X") for z in atomic_numbers]

    with open(filename, "w") as f:
        for i, (sym, (x, y, z)) in enumerate(zip(symbols, pos.tolist()), start=1):
            f.write(f"ATOM  {i:5d} {sym:>2s} MOL     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
        
        bonds = set()
        bond_dict = top.get_arity(2)
        for key in bond_dict:
            if key != "LJ" and key != "coulomb" and key != "LJ_Fudge" and key != "coulomb_Fudge":
                pairs = bond_dict[key]
                for i, j in pairs:
                    # Make sure it's 1-based
                    a, b = i + 1, j + 1
                    bonds.add((min(a, b), max(a, b)))

        for a, b in sorted(bonds):
            f.write(f"CONECT{a:5d}{b:5d}\n")

        f.write("END\n")


def build_top_and_features(prmtop_file: str, *, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32):
    """
    Loads a prmtop file and constructs the topology, node features, and masses.

    Parameters
    ----------
    prmtop_file : str
        Path to the .prmtop file

    Returns
    -------
    top : Topology
        The constructed topology with bonded and nonbonded edges.
    node_features : dict[str, torch.Tensor]
        Dictionary containing per-atom features (sigma, epsilon, charge).
    masses : torch.Tensor
        Tensor of shape (N,) containing atomic masses.
    """
    parm = pmd.load_file(prmtop_file)

    # --- Atom-wise features ---
    sigmas   = torch.tensor([a.sigma for a in parm.atoms], dtype=dtype,device=device)
    epsilons = torch.tensor([a.epsilon for a in parm.atoms], dtype=dtype,device=device)
    charges  = torch.tensor([a.charge for a in parm.atoms], dtype=dtype,device=device)/0.05487686460574314 # convert to my akma derived
    masses   = torch.tensor([a.mass for a in parm.atoms], dtype=dtype,device=device)

    node_features = {
        "sigma": sigmas,
        "epsilon": epsilons,
        "charge": charges,
    }

    # --- Type mappings ---
    bond_type_map     = {bt: f"bondtype_{i}"     for i, bt in enumerate(parm.bond_types)}
    angle_type_map    = {at: f"angletype_{i}"    for i, at in enumerate(parm.angle_types)}
    dihedral_type_map = {dt: f"dihtype_{i}"      for i, dt in enumerate(parm.dihedral_types)}

    labels  = list(bond_type_map.values()) +["LJ","coulomb", "LJ_Fudge","coulomb_Fudge"]  +list(dihedral_type_map.values()) + list(angle_type_map.values())
    arities = [2] * (len(bond_type_map) + 4)  + [4] * len(dihedral_type_map) +  [3] * len(angle_type_map)

    top = Topology((labels, arities),dtype=dtype,device=device)
    energy_dict = {}
    bonded_pairs = set()
    angle_pairs = set()
    dihedral_pairs = set()
    
    # --- Add bonded terms ---
    for b in parm.bonds:
        label = bond_type_map[b.type]
        top.add(2, label, (b.atom1.idx, b.atom2.idx))

        if label not in energy_dict:
            r0 = b.type.req
            kappa = b.type.k
            energy_dict[label] = Harmonic_Bond(r_0=r0, kappa=kappa, label=label,dtype=dtype,device=device)

        bonded_pairs.add(tuple(sorted((b.atom1.idx, b.atom2.idx))))   # 1-2

    for a in parm.angles:
        label = angle_type_map[a.type]
        top.add(3, label, (a.atom1.idx, a.atom2.idx, a.atom3.idx))
        if label not in energy_dict:
            theta0 = a.type.theteq
            theta0_rad = torch.tensor(theta0 * torch.pi / 180.0, device=device, dtype=dtype)
            kappa = a.type.k
            energy_dict[label] = Harmonic_Angle(theta_0=theta0_rad, kappa=kappa, label=label,dtype=dtype,device=device)

        angle_pairs.add(tuple(sorted((a.atom1.idx, a.atom3.idx))))    # 1-3

    for d in parm.dihedrals:
        label = dihedral_type_map[d.type]
        top.add(4, label, (d.atom1.idx, d.atom2.idx, d.atom3.idx, d.atom4.idx))
        if label not in energy_dict:
            phi0_deg = d.type.phase      # degrees
            phi0_rad = torch.tensor(phi0_deg * torch.pi / 180.0, device=device, dtype=dtype)
            kappa    = torch.tensor(d.type.phi_k, device=device, dtype=dtype)
            n        = torch.tensor(d.type.per, device=device, dtype=dtype)
            energy_dict[label] = Dihedral(phi_0=phi0_rad, kappa=kappa, n=n, label=label, dtype=dtype, device=device)

        dihedral_pairs.add(tuple(sorted((d.atom1.idx, d.atom4.idx))))    # 1-4

    # single set of exclusions
    excluded_pairs = bonded_pairs | angle_pairs | dihedral_pairs
    fudge_pairs = dihedral_pairs - bonded_pairs - angle_pairs 
    
    # --- Add all-pair 2-body edges (skip 1-2 and 1-3) --------------------------
    N = len(parm.atoms)
    for i in range(N):
        for j in range(i):
            if not((j, i) in excluded_pairs):          # sorted key matches our storage skip 1-2 / 1-3 / 1-4 pairs
                top.add(2, "LJ",      (i, j))
                top.add(2, "coulomb", (i, j))
            elif (j,i) in fudge_pairs:
                top.add(2, "LJ_Fudge",      (i, j))
                top.add(2, "coulomb_Fudge", (i, j))
                
    energy_dict["coulomb"] = MixtureCoulomb(1,"coulomb",dtype=dtype,device=device)
    energy_dict["LJ"] = MixtureLJ(1,"LJ",dtype=dtype,device=device)
    energy_dict["coulomb_Fudge"] = MixtureCoulomb(1/1.2,"coulomb_Fudge",dtype=dtype,device=device)
    energy_dict["LJ_Fudge"] = MixtureLJ(1/2,"LJ_Fudge",dtype=dtype,device=device)

    return top, node_features, masses, energy_dict

def compute_dihedrals(
    positions: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Compute signed dihedral angles from 4-atom indices and optionally shift them.

    Parameters
    ----------
    positions : torch.Tensor
        Tensor of shape (T, B, N, 3), atomic positions over time and batch.
    indices : torch.Tensor
        LongTensor of shape (M, 4), indices of atoms forming each dihedral.
    Returns
    -------
    phi : torch.Tensor
        Dihedral angles (T, B, M) in radians, in range [-π, π] + phase
    """
    # Gather the 4 positions per dihedral
    vecs = positions[:, :, indices]  # (T, B, M, 4, 3)
    r1, r2, r3, r4 = vecs[..., 0, :], vecs[..., 1, :], vecs[..., 2, :], vecs[..., 3, :]

    # Bond vectors
    b1 = r2 - r1
    b2 = r3 - r2
    b3 = r4 - r3

    # Normal vectors to the planes
    n1 = torch.nn.functional.normalize(torch.cross(b1, b2, dim=-1), dim=-1)
    n2 = torch.nn.functional.normalize(torch.cross(b2, b3, dim=-1), dim=-1)
    b2n = torch.nn.functional.normalize(b2, dim=-1)

    # Compute cosine and sine of the dihedral
    cos_phi = (n1 * n2).sum(dim=-1).clamp(-1.0, 1.0)
    sin_phi = (torch.cross(n1, n2, dim=-1) * b2n).sum(dim=-1)

    # Return signed angle with optional phase shift
    return torch.atan2(sin_phi, cos_phi)

def make_sc_lattice(a: float,
                    nx: int, ny: int, nz: int,
                    *,
                    batch: int = 1,
                    device: str | torch.device = "cuda",
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Simple-cubic lattice duplicated along a batch dimension.

    Returns
    -------
    pos : (B, N, 3) tensor, N = nx * ny * nz
    """
    # integer grid ➜ (nx, ny, nz, 3)
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(nx, device=device, dtype=dtype),
            torch.arange(ny, device=device, dtype=dtype),
            torch.arange(nz, device=device, dtype=dtype),
            indexing="ij"),
        dim=-1)

    # scale and flatten ➜ (N, 3)
    cell = (grid * a).reshape(-1, 3)

    # duplicate along batch ➜ (B, N, 3)
    pos = cell.unsqueeze(0).expand(batch, -1, -1).contiguous()
    return pos

def random_rotation_3d(x: torch.Tensor, R: torch.Tensor = None):
    """
    Apply a 3D rotation to the last dimension of a tensor of shape [B, N, 3].

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape [B, N, 3]
    R : torch.Tensor, optional
        Rotation matrix of shape [3, 3]. If None, a random rotation is generated.

    Returns
    -------
    x_rot : torch.Tensor
        Rotated tensor of shape [B, N, 3]
    R : torch.Tensor
        Rotation matrix used, shape [3, 3]
    R_inv : torch.Tensor
        Inverse of the rotation matrix, shape [3, 3]
    """
    if R is None:
        # Generate a random rotation matrix using QR decomposition
        q, _ = torch.linalg.qr(torch.randn(3, 3, device=x.device, dtype=x.dtype))
        R = q @ torch.diag(torch.sign(torch.linalg.det(q)).expand(3))  # ensure det(R) = +1

    x_rot = torch.matmul(x, R.T)
    R_inv = R.T  # since R is orthogonal, R^-1 = R^T

    return x_rot, R, R_inv