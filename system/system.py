import torch
from torch.func import grad
from system.topology import Topology
from system.box import Box
from system.units import UnitSystem
import random

spark_quotes = [
    "Zero-point energy? Never heard of her.",
    "If you're not shadowing the Hamiltonian, are you really integrating?",
    "My thermostat is the delete key.",
    "Who needs a neighbor list when you have vibes?",
    "These aren’t numerical artifacts or bugs, they’re features.",
    "What do you mean the timestep matters?",
    "The partition function is implicit. Trust the process.",
    "My loss function is the Helmholtz free energy. No notes.",
    "What do you mean KL divergence isn't a potential?",
    "Who needs detailed balance when you’ve got backprop?",
    "My optimizer doesn’t converge—it anneals spiritually.",
    "The gradient didn’t vanish. It ascended.",
    "The phase transition isn’t discontinuous, you just need better resolution.",
    "If it looks non-analytic, zoom harder.",
    "What do you mean I can't use a Taylor series?",
    "My simulation is reversible in spirit.",
    "My free energy surface is locally honest.",
    "I didn’t impose detailed balance. I suggested it gently.",
    "I’m pretty sure the autocorrelation time doesn’t matter here.",
    "The boundary conditions I've got totally don’t affect the bulk properties.",
    "Of course I randomized the seed.",
    "The entropy is increasing. That’s how I know it’s working.",
    "I promise measure theory is helpful to us chemists.",
    "This isn’t overfitting, it’s an infinite-dimensional basis expansion.",
    "We’re not losing resolution! we’re gaining abstraction!",
    "What ensemble even is this?",
    "It’s ergodic in the limit of infinite funding.",
    "I didn’t *forget* the 1-4 scaling. I just ignored it.",
    "I ran the same simulation twice. That’s means its reproducible? Right?",
    "It’s in the canonical ensemble. Ish.",
    "Statistical mechanics: because solving the actual dynamics is too hard.",
    "Stat mech is just Bayesian inference with a heat bath.",
    "The Hamiltonian doesn’t need to be real. Just convincing.",
    "Boltzmann didn’t die for this.",
    "My gas certainly isn't ideal..."
]

spark_banner = f"""
╔═══════════════════════════════════════════════════╗
║                                                   ║
║  ██████╗   ██████╗    ██╗      ██████╗   ██╗  ██╗ ║
║ ██╔════╝  ██╔══██╗   ██╔██╗    ██╔══██╗  ██║ ██╔╝ ║
║ ╚█████╗   ██████╔╝  ██╔╝╚██╗   ██████╔╝  █████╔╝  ║
║  ╚═══██╗  ██╔═══╝  ██╔╝  ╚██╗  ██╔══██╗  ██╔═██╗  ║
║ ██████╔╝  ██║     ██╔╝    ╚██╗ ██║  ██║  ██║ ╚██╗ ║
║ ╚═════╝   ╚═╝     ╚═╝      ╚═╝ ╚═╝  ╚═╝  ╚═╝  ╚═╝ ║
║                                                   ║
║     Statistical Physics Autodiff Research Kit     ║
╚═══════════════════════════════════════════════════╝

          V(r)           ψ, φ              q
           │               │               │
           ○               ○               ○
         ╱ | ╲           ╱ | ╲           ╱ | ╲
        ○  ○  ○         ○  ○  ○         ○  ○  ○
         ╲ | ╱           ╲ | ╱           ╲ | ╱
           ○               ○               ○
           │               │               │
          g(r)             F              E(q)

{random.choice(spark_quotes)}
"""

print(spark_banner)


class System:
    def __init__(self, pos: torch.Tensor, mom: torch.Tensor, mass: torch.Tensor, top: Topology, box: Box, energy_dict: dict, units: UnitSystem, node_features: dict[str, torch.Tensor] = None, *, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32):
        """
        Initialize a batched molecular dynamics system.
        
        Parameters
        ----------
        pos : torch.Tensor
            Atomic positions of shape (B, N, 3), where B is batch size and N is number of atoms.
        mom : torch.Tensor
            Atomic momenta of shape (B, N, 3), same shape as `pos`.
        mass : torch.Tensor
            Atomic masses of shape (N,), assumed identical across batches.
        top : Topology
            Topological information (e.g., bonds, angles) for the system.
        box : Box
            Simulation box defining boundaries.
        energy_dict : dict
            Dictionary mapping interaction labels (str) to energy term objects (each with `.energy(pos, top)`).
        units : UnitSystem
            Unit system defining the physical scale (e.g., kcal/mol, angstrom, fs).
        node_features : dict[str, torch.Tensor]
            Per-atom features of shape (N, ...).
        """
    
        # --- shape checks -------------------------------------------------
        if pos.shape != mom.shape or pos.ndim != 3 or pos.shape[-1] != 3:
            raise ValueError("pos and mom must both be (B, N, 3)")
        if mass.ndim != 1:
            raise ValueError("mass must be (N)")
        B, N = pos.shape[:2]
        mass = mass.unsqueeze(0).expand(B, -1) # expand across batching dim
    
        self.labels = [label for arity in top.edges for label in top.edges[arity]]
        num_labels = len(self.labels)
    
        if len(energy_dict) != num_labels:
            raise ValueError(f"energy_dict has {len(energy_dict)} entries, but {num_labels} labels were found in topology. Every edge type in the hypergraph must have an energy function.")
        
        # --- node features handling ---------------------------------------
        if node_features is not None:
            # Ensure every node has the feature 
            for k, v in node_features.items():
                if v.shape[0] != N:
                    raise ValueError(f"node_feature[{k}] has shape {v.shape}, expected ({N}, ...)")
            self.node_features = {
                k: v.unsqueeze(0).expand(B, *v.shape) for k, v in node_features.items()
            }
        else:
            self.node_features = {}

        # --- store --------------------------------------------------------
        self.pos   = pos
        self.mom   = mom
        self.mass  = mass
        self.box   = box
        self.units = units
        self.top = top
        self.energy_dict = energy_dict
        self._force_fn = None
        self.device = torch.device(device)
        self.dtype = dtype
    
    # --- energy & forces --------------------------------------------------------
    def _potential_energy(self, pos: torch.Tensor) -> torch.Tensor:
        """PRIVATE: batched potential energy at arbitrary `pos` (shape: B,)."""
        return sum((self.energy_dict[label].energy(pos, self.top, self.box, self.node_features)).sum(dim=1) for label in self.labels)

    def potential_energy_split(self) -> torch.Tensor:
        """Potential energy of the current coordinates split by type in dict."""
        return {label:(self.energy_dict[label].energy(self.pos, self.top, self.box, self.node_features)).sum(dim=1) for label in self.labels}
    
    def potential_energy(self) -> torch.Tensor:
        """Public: potential energy of the current coordinates."""
        return self._potential_energy(self.pos)
    
    def compile_force_fn(self):
        """Create a pure, compiled partial E / partial pos callable."""
        self.potential_energy()
        pe_scalar = lambda p: self._potential_energy(p).sum()
        self._force_fn = torch.compile(grad(pe_scalar), fullgraph=True)

    def force(self) -> torch.Tensor:
        """Return - nabla E at current positions."""
        if self._force_fn is None:
            raise RuntimeError("Call compile_force_fn() first.")
        return -self._force_fn(self.pos)

    # --- observables --------------------------------------------------------
    def velocity(self):
        return self.mom / self.mass.unsqueeze(-1) # unsqueze along last dim of mass to make / work

    def kinetic_energy(self):
        return torch.sum(0.5 * self.mom.pow(2) / self.mass.unsqueeze(-1),dim=(1,2)) # don't sum over batch

    def temperature(self):
        return 2*self.kinetic_energy()/(3*self.pos.shape[1]*self.units.kB)

    # --- misc --------------------------------------------------------
    def __repr__(self):
        B, N, _ = self.pos.shape
        L = len(self.labels)
        return (
            f"System(Batches: {B}, Atoms: {N}, Interactions: {L}, "
            f"Box: {self.box}, Units: {self.units})"
        )