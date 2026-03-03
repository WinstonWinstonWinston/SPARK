"""
plot_results.py
---------------
Loads ti_results.npz and crooks_results.npz produced by einstein_nvt.py
and einstein_nvt_jarzynski.py, then produces a combined comparison figure.

Layout (2 rows × N_alpha columns):
  Row 0 : Crooks work distributions per alpha  (P_F and P_R overlaid)
  Row 1 : TI cumulative free-energy path | TI integrand | a/N summary comparison
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -----------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------
dir_path = os.path.dirname(os.path.abspath(__file__))

ti_path  = os.path.join(dir_path, "ti_results.npz")
crk_path = os.path.join(dir_path, "crooks_results.npz")

if not os.path.exists(ti_path):
    sys.exit(f"ERROR: {ti_path} not found — run einstein_nvt.py first.")
if not os.path.exists(crk_path):
    sys.exit(f"ERROR: {crk_path} not found — run einstein_nvt_jarzynski.py first.")

ti  = np.load(ti_path,  allow_pickle=False)
crk = np.load(crk_path, allow_pickle=False)

# shared metadata (use TI file as reference for EOS; rho* identical)
alpha_values    = ti["alpha_values"].tolist()
N_alpha         = len(alpha_values)
lam_np          = ti["lam_np"]
eos_aabs        = float(ti["eos_aabs_per_atom"][0])
T_ti            = float(ti["T"][0])
N_atoms_ti      = int(ti["N_atoms"][0])
L_ti            = float(ti["L"][0])
rho_star_ti     = float(ti["rho_star"][0])
steps_prod      = int(ti["steps_prod"][0])

N_atoms_crk     = int(crk["N_atoms"][0])
L_crk           = float(crk["L"][0])
n_passes        = int(crk["n_passes"][0])
steps_switch    = int(crk["steps_switch"][0])

colors = plt.cm.tab10.colors

# -----------------------------------------------------------------------
# Build figure:  2 rows × N_alpha cols, with the summary in row-1 col 2
# -----------------------------------------------------------------------
fig = plt.figure(figsize=(5 * N_alpha, 9))
gs  = gridspec.GridSpec(2, N_alpha, figure=fig, hspace=0.45, wspace=0.35)

# --- Row 0: Crooks work distributions ---
for idx, alpha in enumerate(alpha_values):
    pfx = f"a{int(alpha)}_"
    ax  = fig.add_subplot(gs[0, idx])
    c   = colors[idx % len(colors)]

    w_grid   = crk[pfx+"w_grid"]
    p_f      = crk[pfx+"p_f"]
    p_r      = crk[pfx+"p_r"]
    dF_crooks = float(crk[pfx+"dF_crooks"][0])
    dF_target = float(crk[pfx+"dF_target"][0])
    a_est     = float(crk[pfx+"a_abs_est"][0])

    ax.plot(w_grid, p_f, color=c, lw=2.0, label=r"$P_F(W)$")
    ax.plot(w_grid, p_r, color=c, lw=2.0, ls="--", label=r"$P_R(-W)$")
    ax.axvline(dF_crooks, color="black", lw=1.5,
               label=rf"Crooks: {dF_crooks:.2f}")
    ax.axvline(dF_target, color="gray", lw=1.5, ls=":",
               label=rf"EOS: {dF_target:.2f}")
    ax.set_title(
        rf"Crooks  $\alpha$={alpha:.0f}"
        + "\n"
        + rf"$a/N$={a_est:.4f}  |  $n_{{traj}}$={len(crk[pfx+'W_fwd'])}",
        fontsize=9,
    )
    ax.set_xlabel("Work $W$")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)

# --- Row 1 col 0: TI cumulative free-energy path ---
ax_fe = fig.add_subplot(gs[1, 0])
for idx, alpha in enumerate(alpha_values):
    pfx = f"a{int(alpha)}_"
    c   = colors[idx % len(colors)]
    fp  = ti[pfx+"F_path_per_atom_abs"]
    F_LJ = float(ti[pfx+"F_LJ"][0])
    ax_fe.plot(lam_np, fp, marker="o", ms=1.5, lw=1.5, color=c,
               label=rf"$\alpha$={alpha:.0f}  (a/N={F_LJ/N_atoms_ti:.4f})")
ax_fe.axhline(eos_aabs, color="gray", ls="--",
              label=rf"EOS: {eos_aabs:.4f}")
ax_fe.set_xlabel(r"$\lambda$")
ax_fe.set_ylabel("Absolute free energy per atom")
ax_fe.set_title("TI cumulative free-energy path", fontsize=9)
ax_fe.legend(fontsize=7)

# --- Row 1 col 1: TI integrand ---
ax_int = fig.add_subplot(gs[1, 1])
for idx, alpha in enumerate(alpha_values):
    pfx = f"a{int(alpha)}_"
    c   = colors[idx % len(colors)]
    ax_int.plot(lam_np, ti[pfx+"mean_diff"], marker="o", ms=1.5, lw=1.5,
                color=c, label=rf"$\alpha$={alpha:.0f}")
ax_int.set_xlabel(r"$\lambda$")
ax_int.set_ylabel(r"$\langle U_{LJ} - U_{EC} \rangle_\lambda$")
ax_int.set_title("TI integrand", fontsize=9)
ax_int.legend(fontsize=7)

# --- Row 1 col 2: summary a/N comparison ---
ax_sum = fig.add_subplot(gs[1, 2])

x      = np.arange(N_alpha)
width  = 0.18
labels = [rf"$\alpha$={int(a)}" for a in alpha_values]

# TI estimates (last value of cumulative path = λ=1 endpoint)
ti_vals    = [float(ti[f"a{int(a)}_F_path_per_atom_abs"][-1]) for a in alpha_values]
# Crooks crossing estimates
crk_vals   = [float(crk[f"a{int(a)}_a_abs_est"][0]) for a in alpha_values]
# Jarzynski forward
jar_f_vals = [float(crk[f"a{int(a)}_F_LJ_abs_jar_f"][0]) / N_atoms_crk for a in alpha_values]
# Jarzynski reverse
jar_r_vals = [float(crk[f"a{int(a)}_F_LJ_abs_jar_r"][0]) / N_atoms_crk for a in alpha_values]

ax_sum.bar(x - 1.5*width, ti_vals,    width, label="TI",          color="steelblue",  alpha=0.85)
ax_sum.bar(x - 0.5*width, crk_vals,   width, label="Crooks",      color="darkorange", alpha=0.85)
ax_sum.bar(x + 0.5*width, jar_f_vals, width, label="Jar. fwd",    color="seagreen",   alpha=0.85)
ax_sum.bar(x + 1.5*width, jar_r_vals, width, label="Jar. rev",    color="orchid",     alpha=0.85)
ax_sum.axhline(eos_aabs, color="black", ls="--", lw=1.5,
               label=rf"EOS: {eos_aabs:.4f}")

ax_sum.set_xticks(x)
ax_sum.set_xticklabels(labels, fontsize=8)
ax_sum.set_ylabel("$a/N$ (absolute free energy per atom)")
ax_sum.set_title("Method comparison", fontsize=9)
ax_sum.legend(fontsize=7)

# -----------------------------------------------------------------------
# Supertitle and save
# -----------------------------------------------------------------------
fig.suptitle(
    rf"LJ solid free energy  |  $\rho^*$={rho_star_ti:.4f}, $T^*$={T_ti:.2f}"
    "\n"
    rf"TI: N={N_atoms_ti}, B={int(ti['B'][0])}, steps_prod={steps_prod}"
    rf"   |   Crooks: N={N_atoms_crk}, B={int(crk['B'][0])}, "
    rf"n_passes={n_passes}, steps_switch={steps_switch}",
    fontsize=10,
)

out = os.path.join(dir_path, "comparison_results.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
plt.show()
