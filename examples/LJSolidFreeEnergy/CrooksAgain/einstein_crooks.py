import os

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import spark.system.units as units
import spark.system.topology as topology
import spark.system.box as box
import spark.system.system as sys_module

from spark.forces.onebody import EinsteinCrystal
from spark.forces.twobody import Mie
from spark.forces.composite import InterpolatingPotential
from spark.integrators.NVT import NVT
from spark.utils import make_fcc_lattice

# ---------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--alpha",        type=float, required=True,
                    choices=[150.0, 200.0, 250.0])
parser.add_argument("--schedule",     type=str,   required=True,
                    choices=["cosine", "linear"])
parser.add_argument("--steps_switch", type=int,   default=2000)
parser.add_argument("--steps_eq",     type=int,   default=2000)
args = parser.parse_args()

alpha        = args.alpha
schedule     = args.schedule
steps_switch = args.steps_switch
steps_eq     = args.steps_eq

device = "cuda"
dtype  = torch.float32

# ---------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------
a          = 1.54174
nx, ny, nz = 7, 7, 7
L          = a * nx
B          = 256
T          = 1.0
dt         = 0.005
gamma      = 10.0
n_passes   = 10
print_every = 100

# lambda schedule
if schedule == "cosine":
    s = torch.linspace(0.0, 1.0, steps_switch + 1, device=device, dtype=dtype)
    lam_schedule = 0.5 * (1.0 - torch.cos(torch.pi * s))
else:
    lam_schedule = torch.linspace(0.0, 1.0, steps_switch + 1, device=device, dtype=dtype)

# ---------------------------------------------------------------------
# Build FCC lattice and topology once
# ---------------------------------------------------------------------
pos_init = make_fcc_lattice(a=a, nx=nx, ny=ny, nz=nz, batch=B, device=device, dtype=dtype)
pos_init = pos_init - pos_init.mean(dim=1, keepdim=True) + \
           torch.tensor([[L, L, L]], device=device, dtype=dtype) / 2
R       = pos_init[0].clone()
N_atoms = pos_init.shape[1]

b   = box.Box([L, L, L], ["p", "p", "p"], device=device, dtype=dtype)
top = topology.Topology((["mie"], [2]), device=device, dtype=dtype)
for i in range(N_atoms):
    for j in range(i):
        top.add(2, "mie", (i, j))

mass = torch.ones(N_atoms, device=device, dtype=dtype)
u    = units.UnitSystem()

# ---------------------------------------------------------------------
# van der Hoef EOS (optional reference target)
# ---------------------------------------------------------------------
rho_star = N_atoms / (L ** 3)

def vdh_fcc_lj_beta_aex(rho, T):
    beta = 1.0 / T
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

eos_beta_aex      = vdh_fcc_lj_beta_aex(rho_star, T)
eos_aex_per_atom  = eos_beta_aex * T
eos_aid_per_atom  = T * (np.log(rho_star) - 1.0)   # reduced units: Lambda = 1
eos_aabs_per_atom = eos_aex_per_atom + eos_aid_per_atom

F_LJ_target_abs = N_atoms * eos_aabs_per_atom
F_LJ_target_cm  = F_LJ_target_abs + T * np.log(L**3)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def make_lambda_feature(lam):
    return {"lambda": torch.full((B, N_atoms), float(lam), device=device, dtype=dtype)}

def set_lambda(S, lam):
    S.node_features["lambda"].fill_(float(lam))

def build_system(alpha, lam0):
    pos = pos_init.clone()
    mom = np.sqrt(T) * torch.randn(B, N_atoms, 3, device=device, dtype=dtype)
    einstein = EinsteinCrystal(R=R, alpha=alpha, label="einstein", device=device, dtype=dtype)
    mie      = Mie(n=12, sigma=1.0, epsilon=1.0, label="mie", device=device, dtype=dtype)
    interp   = InterpolatingPotential(einstein, mie, label="interp", device=device, dtype=dtype)
    energy_dict   = {"interp": interp}
    node_features = make_lambda_feature(lam0)
    S = sys_module.System(
        pos, mom, mass, top, b, energy_dict, u,
        node_features=node_features, device=device, dtype=dtype
    )
    S.compile_force_fn()
    integrator = NVT(
        dt=dt, gamma=gamma, T=T,
        remove_com_mom=True,
        remove_com_pos=True,
        device=device, dtype=dtype
    )
    return S, integrator

def U_at_lambda(S, lam):
    set_lambda(S, lam)
    S.reset_cache()
    return S.potential_energy()   # (B,)

def equilibrate(S, integrator, steps, desc):
    with torch.no_grad():
        for _ in tqdm(range(steps), desc=desc, leave=False):
            integrator.step(S)
            S.reset_cache()

def run_switch(S, integrator, lam_sched, desc):
    works = torch.zeros(B, device=device, dtype=dtype)
    set_lambda(S, lam_sched[0].item())
    S.reset_cache()
    with torch.no_grad():
        pbar = tqdm(range(len(lam_sched) - 1), desc=desc, leave=False)
        for k in pbar:
            lam0 = lam_sched[k].item()
            lam1 = lam_sched[k + 1].item()
            # protocol work at fixed x_k
            U0 = U_at_lambda(S, lam0)
            U1 = U_at_lambda(S, lam1)
            works += U1 - U0
            # propagate at new lambda
            set_lambda(S, lam1)
            S.reset_cache()
            integrator.step(S)
            S.reset_cache()
            if k % print_every == 0:
                pbar.set_postfix(
                    lam=f"{lam1:.3f}",
                    U0=f"{U0.mean().item():.2f}",
                    U1=f"{U1.mean().item():.2f}",
                    dW=f"{(U1 - U0).mean().item():.2f}",
                    W=f"{works.mean().item():.2f}",
                    T=f"{S.temperature().mean().item():.3f}",
                )
    return works.cpu().numpy()

def logmeanexp(x):
    x = np.asarray(x, dtype=np.float64)
    xmax = np.max(x)
    return xmax + np.log(np.mean(np.exp(x - xmax)))

def kde_1d(samples, grid):
    x = np.asarray(samples, dtype=float)
    n = len(x)
    std = np.std(x, ddof=1) if n > 1 else 1.0
    iqr = np.subtract(*np.percentile(x, [75, 25])) if n > 1 else 0.0
    sigma = min(std, iqr / 1.349) if iqr > 0 else std
    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = max(std, 1e-3)
    bw = max(0.9 * sigma * n ** (-1 / 5), 1e-3)
    z = (grid[:, None] - x[None, :]) / bw
    pdf = np.exp(-0.5 * z * z).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
    return pdf

def crooks_crossing(W_fwd, W_rev, n_grid=4000):
    x_f = np.asarray(W_fwd, dtype=float)
    x_r = -np.asarray(W_rev, dtype=float)
    lo = min(x_f.min(), x_r.min())
    hi = max(x_f.max(), x_r.max())
    pad = 0.15 * max(hi - lo, 1.0)
    grid = np.linspace(lo - pad, hi + pad, n_grid)
    p_f = kde_1d(x_f, grid)
    p_r = kde_1d(x_r, grid)
    diff = p_f - p_r
    idx = np.where(diff[:-1] * diff[1:] <= 0)[0]
    if len(idx) > 0:
        target = 0.5 * (x_f.mean() + x_r.mean())
        mids = 0.5 * (grid[idx] + grid[idx + 1])
        i = idx[np.argmin(np.abs(mids - target))]
        x0, x1 = grid[i], grid[i + 1]
        y0, y1 = diff[i], diff[i + 1]
        w_cross = x0 - y0 * (x1 - x0) / (y1 - y0) if y1 != y0 else 0.5 * (x0 + x1)
    else:
        w_cross = grid[np.argmin(np.abs(diff))]
    return w_cross, grid, p_f, p_r

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  alpha={alpha}  schedule={schedule}  steps_switch={steps_switch}  steps_eq={steps_eq}")
print(f"{'='*60}")

F_EC = T * (1.5 * np.log(N_atoms) - 1.5 * (N_atoms - 1) * np.log(2 * np.pi * T / alpha))
dF_target = F_LJ_target_cm - F_EC

all_W_fwd = []
all_W_rev = []

pass_bar = tqdm(range(n_passes), desc=f"passes α={alpha}")
for p in pass_bar:
    # forward: EC -> LJ
    S_fwd, integrator_fwd = build_system(alpha, lam0=0.0)
    equilibrate(S_fwd, integrator_fwd, steps_eq, desc=f"  Equil F α={alpha} pass={p+1}")
    W_fwd = run_switch(S_fwd, integrator_fwd, lam_schedule, desc=f"  Fwd α={alpha} pass={p+1}")
    all_W_fwd.append(W_fwd)
    # reverse: LJ -> EC
    S_rev, integrator_rev = build_system(alpha, lam0=1.0)
    equilibrate(S_rev, integrator_rev, steps_eq, desc=f"  Equil R α={alpha} pass={p+1}")
    W_rev = run_switch(S_rev, integrator_rev, torch.flip(lam_schedule, dims=[0]),
                       desc=f"  Rev α={alpha} pass={p+1}")
    all_W_rev.append(W_rev)
    Wf_cat = np.concatenate(all_W_fwd)
    Wr_cat = np.concatenate(all_W_rev)
    dF_live, _, _, _ = crooks_crossing(Wf_cat, Wr_cat)
    pass_bar.set_postfix(
        ntraj=len(Wf_cat),
        dF=f"{dF_live:.2f}",
        Wf=f"{Wf_cat.mean():.2f}",
        Wr=f"{Wr_cat.mean():.2f}",
    )

W_fwd = np.concatenate(all_W_fwd)
W_rev = np.concatenate(all_W_rev)

dF_crooks, w_grid, p_f, p_r = crooks_crossing(W_fwd, W_rev)

# stable Jarzynski diagnostics
dF_jar_f = -T * logmeanexp(-W_fwd / T)
dF_jar_r =  T * logmeanexp(-W_rev / T)

F_LJ_cm_est  = F_EC + dF_crooks
F_LJ_abs_est = F_LJ_cm_est - T * np.log(L**3)
a_abs_est    = F_LJ_abs_est / N_atoms

print(f"\n--- Crooks Estimate (alpha={alpha}) ---")
print(f"N atoms                = {N_atoms}")
print(f"n_passes               = {n_passes}")
print(f"trajectories / dir     = {len(W_fwd)}")
print(f"steps_switch           = {steps_switch}")
print(f"schedule               = {schedule}")
print(f"F_EC (COM-constrained) = {F_EC:.6f}")
print(f"ΔF target (EOS, CM)    = {dF_target:.6f}")
print(f"ΔF Crooks crossing     = {dF_crooks:.6f}")
print(f"ΔF Jarzynski forward   = {dF_jar_f:.6f}")
print(f"ΔF Jarzynski reverse   = {dF_jar_r:.6f}")
print(f"F_LJ abs estimate      = {F_LJ_abs_est:.6f}")
print(f"F_LJ/N estimate        = {a_abs_est:.6f}")
print(f"EOS a_ex/N             = {eos_aex_per_atom:.6f}")
print(f"EOS a_id/N             = {eos_aid_per_atom:.6f}")
print(f"EOS a/N                = {eos_aabs_per_atom:.6f}")
print(f"error in a/N           = {a_abs_est - eos_aabs_per_atom:.6f}")

# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------
tag     = f"alpha{int(alpha)}_{schedule}_sw{steps_switch}_eq{steps_eq}"
out_npz = os.path.join(os.path.dirname(__file__), f"crooks_results_{tag}.npz")
save_dict = {
    "alpha":             np.array([alpha]),
    "schedule":          np.array([schedule]),
    "eos_aabs_per_atom": np.array([eos_aabs_per_atom]),
    "eos_aex_per_atom":  np.array([eos_aex_per_atom]),
    "eos_aid_per_atom":  np.array([eos_aid_per_atom]),
    "rho_star":          np.array([rho_star]),
    "N_atoms":           np.array([N_atoms]),
    "T":                 np.array([T]),
    "L":                 np.array([L]),
    "B":                 np.array([B]),
    "n_passes":          np.array([n_passes]),
    "steps_switch":      np.array([steps_switch]),
    "steps_eq":          np.array([steps_eq]),
    "W_fwd":             W_fwd,
    "W_rev":             W_rev,
    "dF_crooks":         np.array([dF_crooks]),
    "dF_jar_f":          np.array([dF_jar_f]),
    "dF_jar_r":          np.array([dF_jar_r]),
    "dF_target":         np.array([dF_target]),
    "F_EC":              np.array([F_EC]),
    "F_LJ_abs_est":      np.array([F_LJ_abs_est]),
    "a_abs_est":         np.array([a_abs_est]),
    "w_grid":            w_grid,
    "p_f":               p_f,
    "p_r":               p_r,
}
np.savez(out_npz, **save_dict)
print(f"\nSaved results -> {out_npz}")

# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
c = plt.cm.tab10.colors[0]

Wf = np.asarray(W_fwd)
Wr = -np.asarray(W_rev)

wmin = min(Wf.min(), Wr.min())
wmax = max(Wf.max(), Wr.max())
bins = np.linspace(wmin, wmax, 50)

ax.hist(Wf, bins=bins, density=True, alpha=0.22, color=c,      label=r"$P_F(W)$ hist")
ax.hist(Wr, bins=bins, density=True, alpha=0.22, color="gray", label=r"$P_R(-W)$ hist")
ax.plot(w_grid, p_f, color=c,       linewidth=2.0,                  label=r"$P_F(W)$")
ax.plot(w_grid, p_r, color="black", linewidth=2.0, linestyle="--",  label=r"$P_R(-W)$")

overlap = np.minimum(p_f, p_r)
ax.fill_between(w_grid, 0.0, overlap, color="green", alpha=0.18, label="overlap")
ax.axvline(dF_crooks, color="red",  linewidth=1.8,
           label=rf"Crooks $\Delta F$ = {dF_crooks:.2f}")
ax.axvline(dF_target, color="gray", linewidth=1.8, linestyle=":",
           label=rf"EOS target = {dF_target:.2f}")

ax.set_title(
    rf"$\alpha$={int(alpha)} | sched={schedule} | $a/N$={a_abs_est:.4f}"
    + "\n"
    + rf"$n_{{\rm traj}}$={len(W_fwd)}, $n_{{\rm sw}}$={steps_switch}"
)
ax.set_xlabel("Work")
ax.set_ylabel("Probability density")
ax.legend(fontsize=7)

fig.suptitle(
    rf"Crooks switching: EC $\leftrightarrow$ LJ | N={N_atoms}, T={T}, "
    rf"$\rho^*$={rho_star:.4f}, B={B}, n_passes={n_passes}, steps_switch={steps_switch}",
    fontsize=11
)
fig.tight_layout()

out = os.path.join(os.path.dirname(__file__), f"crooks_work_distributions_{tag}.png")
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")
