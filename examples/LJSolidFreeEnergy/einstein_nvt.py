import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import system.units as units
import system.topology as topology
import system.box as box
import system.system as sys_module
from forces.onebody import EinsteinCrystal
from forces.twobody import Mie
from forces.composite import InterpolatingPotential
from integrators.NVT import NVT
from utils import make_fcc_lattice

device = "cuda"
dtype  = torch.float32

# --- system parameters ---
a          = 1.54174
nx, ny, nz = 7, 7, 7
L          = a * nx
B          = 256
T          = 1.0
dt         = 0.005
gamma      = 10.0

alpha_values = [150, 200, 250]          # sweep over these spring constants

lam_vals = torch.linspace(1e-5, 1, B, device=device, dtype=dtype)   # (B,)
lam_np   = lam_vals.cpu().numpy()

# --- build FCC lattice once (reset per alpha run) ---
pos_init = make_fcc_lattice(a=a, nx=nx, ny=ny, nz=nz, batch=B, device=device, dtype=dtype)
pos_init = pos_init - pos_init.mean(dim=1, keepdim=True) + \
           torch.tensor([[L, L, L]], device=device, dtype=dtype) / 2
R       = pos_init[0].clone()    # reference positions for Einstein crystal
N_atoms = pos_init.shape[1]

# --- topology (same for all alpha) ---
b   = box.Box([L, L, L], ["p", "p", "p"], device=device, dtype=dtype)
top = topology.Topology((["mie"], [2]), device=device, dtype=dtype)
for i in range(N_atoms):
    for j in range(i):
        top.add(2, "mie", (i, j))

mass = torch.ones(N_atoms, device=device, dtype=dtype)
u    = units.UnitSystem()

node_features = {"lambda": lam_vals.unsqueeze(1).expand(B, N_atoms).contiguous()}

# --- van der Hoef EOS (alpha-independent reference) ---
rho_star = N_atoms / (L ** 3)

def vdh_fcc_lj_beta_aex(rho, T):
    beta = 1.0 / T
    # van der Hoef fit coefficients for fcc LJ solid
    c2, c4 = -14.45392093, 6.065940096
    C = -23.3450759
    a_nm = {
        (0, 2): -8.2151768, (0, 3): 12.0706860, (0, 4): -6.6594615, (0, 5): 1.3211582,
        (1, 2): 13.4040690, (1, 3): -20.6320660, (1, 4): 11.5648250, (1, 5): -2.3064801,
        (2, 2): -5.5481261, (2, 3):  8.8465978, (2, 4): -5.0258631, (2, 5):  1.0070066,
    }
    b_n = {0: 69.833875, 1: -132.86963, 2: 97.438593, 3: -25.848057}
    u_stat = c2 * rho**2 + c4 * rho**4
    U_ah = 0.0
    for n in range(3):
        for m in range(2, 6):
            U_ah += -(a_nm[(n, m)] / (m - 1)) * rho**n * beta**(-m + 1)
    poly = sum((b_n[n] / (n + 1)) * rho**(n + 1) for n in range(4))
    return C + beta * u_stat + 1.5 * np.log(beta) + U_ah + poly

# van der Hoef EOS: excess and total absolute free energy per atom
eos_beta_aex      = vdh_fcc_lj_beta_aex(rho_star, T)
eos_aex_per_atom  = eos_beta_aex * T
eos_aid_per_atom  = T * (np.log(rho_star) - 1.0)   # reduced units: Lambda = 1
eos_aabs_per_atom = eos_aex_per_atom + eos_aid_per_atom

# --- simulation parameters ---
steps_eq    = 1_000
steps_prod  = 5_000
print_every = 100   # postfix update frequency

# -----------------------------------------------------------------------
# Loop over alpha values
# -----------------------------------------------------------------------
results = {}   # alpha -> dict with mean_diff, mean_msd, F_EC, F_LJ

for alpha in alpha_values:
    print(f"\n{'='*60}")
    print(f"  alpha = {alpha}")
    print(f"{'='*60}")

    # --- reset positions and momenta ---
    pos = pos_init.clone()
    mom = (T ** 0.5) * torch.randn(B, N_atoms, 3, device=device, dtype=dtype)

    # --- build forces ---
    einstein = EinsteinCrystal(R=R, alpha=alpha, label="einstein", device=device, dtype=dtype)
    mie      = Mie(n=12, sigma=1.0, epsilon=1.0, label="mie", device=device, dtype=dtype)
    interp   = InterpolatingPotential(einstein, mie, label="mie", device=device, dtype=dtype)

    energy_dict = {"mie": interp}

    # --- build system ---
    S = sys_module.System(pos, mom, mass, top, b, energy_dict, u,
                          node_features=node_features, device=device, dtype=dtype)
    S.compile_force_fn()
    print(S)

    integrator = NVT(dt=dt, gamma=gamma, T=T,
                     remove_com_mom=True, remove_com_pos=False,
                     device=device, dtype=dtype)
    print(integrator)
    print(f"λ range: [{lam_np[0]:.3f}, {lam_np[-1]:.3f}]  B={B}\n")

    # --- equilibration ---
    with torch.no_grad():
        for _ in tqdm(range(steps_eq), desc=f"  Equil  α={alpha}", leave=False):
            integrator.step(S)
            S.reset_cache()

    # --- production ---
    msd_accum  = torch.zeros(B, device=device, dtype=dtype)
    diff_accum = torch.zeros(B, device=device, dtype=dtype)
    counts = 0

    # precompute lam tensor on device for split recovery
    lam_b    = lam_vals                          # (B,)
    inv_lam  = 1.0 / lam_b.clamp(min=1e-10)     # (B,)
    inv_1lam = 1.0 / (1.0 - lam_b).clamp(min=1e-10)  # (B,)

    with torch.no_grad():
        pbar = tqdm(range(steps_prod), desc=f"  Prod   α={alpha}")
        for i in pbar:
            integrator.step(S)   # leaves _cached_force set; _cached_pe_split not yet set

            # one forward pass through InterpolatingPotential (caches the result)
            split   = S.potential_energy_split()              # {"mie": (B, N_atoms + M_pairs)}
            raw     = split["mie"]                            # (B, M_a + M_b)
            # recover unscaled sub-energies by inverting the (1-λ)/λ weighting
            E_ec    = raw[:, :N_atoms].sum(dim=1) * inv_1lam  # (B,)
            E_lj    = raw[:, N_atoms:].sum(dim=1) * inv_lam   # (B,)
            msd     = (S.pos - R.unsqueeze(0)).pow(2).sum(dim=-1).mean(dim=-1)  # (B,)

            S.reset_cache()   # clear after reading split

            diff_accum += E_lj - E_ec
            msd_accum  += msd
            counts     += 1

            if i % print_every == 0:
                pbar.set_postfix(
                    T=f"{S.temperature().mean().item():.3f}",
                    dU0=f"{(E_lj - E_ec)[0].item():.2f}",
                    dU1=f"{(E_lj - E_ec)[-1].item():.2f}",
                )

    mean_diff = (diff_accum / counts).cpu().numpy()
    mean_msd  = (msd_accum  / counts).cpu().numpy()

    # --- free energy estimate via TI ---
    N  = N_atoms
    kT = T
    F_EC  = kT * (1.5 * np.log(N) - 1.5 * (N - 1) * np.log(2 * np.pi * kT / alpha))
    dF_TI = np.trapezoid(mean_diff, lam_np)
    F_LJ  = F_EC + dF_TI

    # --- cumulative TI free energy path ---
    # COM-constrained convention → unconstrained absolute free energy per atom
    cum_dF = np.concatenate((
        [0.0],
        np.cumsum(0.5 * (mean_diff[1:] + mean_diff[:-1]) * np.diff(lam_np))
    ))
    F_path_per_atom_abs = (F_EC + cum_dF - T * np.log(L**3)) / N

    print(f"\n--- Free Energy Estimate (alpha={alpha}) ---")
    print(f"N atoms          = {N}")
    print(f"F_EC (analytical)= {F_EC:.4f}")
    print(f"∫<U_LJ-U_EC>dλ   = {dF_TI:.4f}   [λ: 0 → 1]")
    print(f"F_LJ estimate    = {F_LJ:.4f}")
    print(f"F_LJ/N estimate  = {F_LJ/N:.6f}")
    print(f"van der Hoef EOS a_ex/N  = {eos_aex_per_atom:.6f}")
    print(f"ideal-gas a_id/N         = {eos_aid_per_atom:.6f}")
    print(f"van der Hoef EOS a/N     = {eos_aabs_per_atom:.6f}  at rho*={rho_star:.6f}, T*={T:.3f}")

    results[alpha] = dict(mean_diff=mean_diff, mean_msd=mean_msd,
                          F_EC=F_EC, F_LJ=F_LJ,
                          F_path_per_atom_abs=F_path_per_atom_abs,
                          alpha=alpha)

# -----------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------
out_npz = os.path.join(os.path.dirname(__file__), "ti_results.npz")
save_dict = {
    "alpha_values":      np.array(alpha_values),
    "lam_np":            lam_np,
    "eos_aabs_per_atom": np.array([eos_aabs_per_atom]),
    "eos_aex_per_atom":  np.array([eos_aex_per_atom]),
    "eos_aid_per_atom":  np.array([eos_aid_per_atom]),
    "rho_star":          np.array([rho_star]),
    "N_atoms":           np.array([N_atoms]),
    "T":                 np.array([T]),
    "L":                 np.array([L]),
    "B":                 np.array([B]),
    "steps_prod":        np.array([steps_prod]),
}
for alpha in alpha_values:
    r   = results[alpha]
    pfx = f"a{int(alpha)}_"
    save_dict.update({
        pfx+"mean_diff":          r["mean_diff"],
        pfx+"mean_msd":           r["mean_msd"],
        pfx+"F_EC":               np.array([r["F_EC"]]),
        pfx+"F_LJ":               np.array([r["F_LJ"]]),
        pfx+"F_path_per_atom_abs":r["F_path_per_atom_abs"],
    })
np.savez(out_npz, **save_dict)
print(f"Saved results → {out_npz}")

# -----------------------------------------------------------------------
# Plot all alpha results on the same axes
# -----------------------------------------------------------------------
colors = plt.cm.tab10.colors

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for idx, alpha in enumerate(alpha_values):
    r   = results[alpha]
    c   = colors[idx % len(colors)]
    lbl = rf"$\alpha$={alpha}  (a/N={r['F_LJ']/N_atoms:.3f})"

    # TI integrand
    axes[0].plot(lam_np, r['mean_diff'], marker="o", markersize=2,
                 linewidth=1.5, color=c, label=lbl)

    # MSD  (equipartition reference: 3kT/alpha per atom)
    axes[1].plot(lam_np, r['mean_msd'], marker="o", markersize=2,
                 linewidth=1.5, color=c, label=lbl)
    axes[1].axhline(3 * T / alpha, color=c, linestyle="--", linewidth=0.8,
                    label=rf"Equipart. $\alpha$={alpha}: {3*T/alpha:.4f}")

    # Cumulative TI free-energy path (absolute, per atom)
    axes[2].plot(lam_np, r['F_path_per_atom_abs'], marker="o", markersize=3,
                 linewidth=1.5, color=c, label=lbl)

# van der Hoef EOS reference drawn once (alpha-independent)
axes[2].axhline(
    eos_aabs_per_atom,
    color="gray", linestyle="--",
    label=rf"van der Hoef EOS: $a/N$ = {eos_aabs_per_atom:.4f}"
)

axes[0].set_xlabel(r"$\lambda$")
axes[0].set_ylabel(r"$\langle U_\mathrm{LJ} - U_\mathrm{EC} \rangle_\lambda$")
axes[0].set_title("TI integrand")
axes[0].legend(fontsize=8)

axes[1].set_xlabel(r"$\lambda$")
axes[1].set_ylabel("Mean MSD per atom")
axes[1].set_title("MSD vs λ")
axes[1].legend(fontsize=8)

axes[2].set_xlabel(r"$\lambda$")
axes[2].set_ylabel("Absolute free energy per atom")
axes[2].set_title("Cumulative TI free-energy path")
axes[2].legend(fontsize=8)

fig.suptitle(rf"EC $\to$ LJ  |  N={N_atoms}, T={T}, $\rho^*$={rho_star:.4f}, B={B}, steps_prod={steps_prod}",
             fontsize=11)
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "msd_vs_lambda.png")
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")
