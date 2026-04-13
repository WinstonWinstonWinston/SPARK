"""
Produces separate PNGs saved to ~/SPARK/figs/:

  ti_plot.png                          — TI integrand / MSD / cumulative FE
  bar_comparison.png                   — a/N bar chart (all methods)
  crooks_alpha{A}_{sched}.png  ×6     — histogram (top) + schedule (bottom)
"""
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

HERE   = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.join(HERE, "..")
FIGS   = os.path.join(HERE, "..", "..", "..", "figs")
os.makedirs(FIGS, exist_ok=True)

# =====================================================================
# Load data
# =====================================================================
ti       = np.load(os.path.join(PARENT, "ti_results.npz"),     allow_pickle=True)
outer_cr = np.load(os.path.join(PARENT, "crooks_results.npz"), allow_pickle=True)

ti_alphas = ti["alpha_values"]
lam_np    = ti["lam_np"]
eos_aabs  = float(ti["eos_aabs_per_atom"][0])
N_atoms   = int(ti["N_atoms"][0])
rho_star  = float(ti["rho_star"][0])

cr_files = sorted(glob.glob(os.path.join(HERE, "crooks_results_alpha*.npz")))
cr_runs  = []
for f in cr_files:
    d = np.load(f, allow_pickle=True)
    cr_runs.append({
        "alpha":        float(np.asarray(d["alpha"]).flat[0]),
        "schedule":     str(d["schedule"][0]),
        "steps_switch": int(d["steps_switch"][0]),
        "steps_eq":     int(d["steps_eq"][0]),
        "W_fwd":        np.asarray(d["W_fwd"]),
        "W_rev":        np.asarray(d["W_rev"]),
        "w_grid":       np.asarray(d["w_grid"]),
        "p_f":          np.asarray(d["p_f"]),
        "p_r":          np.asarray(d["p_r"]),
        "dF_crooks":    float(d["dF_crooks"][0]),
        "dF_target":    float(d["dF_target"][0]),
        "a_abs_est":    float(d["a_abs_est"][0]),
        "eos_aabs":     float(d["eos_aabs_per_atom"][0]),
    })
cr_runs.sort(key=lambda r: (r["schedule"], r["alpha"]))

alphas    = sorted(set(r["alpha"] for r in cr_runs))
schedules = ["cosine", "linear"]

# =====================================================================
# Style
# =====================================================================
alpha_colors = {a: c for a, c in zip(alphas, plt.cm.tab10.colors)}
sched_ls     = {"cosine": "-",         "linear": "--"}
sched_col    = {"cosine": "steelblue", "linear": "tomato"}

META = rf"LJ solid | N={N_atoms}, T=1.0, $\rho^*$={rho_star:.4f}, B=256"

# =====================================================================
# 1. TI plot  — top: [TI integrand | cumulative FE],  bottom: MSD
# =====================================================================
fig_ti = plt.figure(figsize=(14, 8))
gs_ti  = gridspec.GridSpec(2, 2, figure=fig_ti,
                            height_ratios=[1.1, 1], hspace=0.38, wspace=0.32)
ax_intg = fig_ti.add_subplot(gs_ti[0, 0])
ax_cfe  = fig_ti.add_subplot(gs_ti[0, 1])
ax_msd  = fig_ti.add_subplot(gs_ti[1, :])   # full-width bottom

T_sim = 1.0   # T used in MD; needed for equipartition line

for alpha in ti_alphas:
    pfx = f"a{int(alpha)}_"
    c   = alpha_colors[alpha]

    mean_diff = ti[pfx + "mean_diff"]
    mean_msd  = ti[pfx + "mean_msd"]
    fpath     = ti[pfx + "F_path_per_atom_abs"]
    dF_ti     = float(np.trapezoid(mean_diff, lam_np)) / N_atoms

    # TI integrand
    ax_intg.plot(lam_np, mean_diff, color=c, lw=2.0,
                 label=rf"$\alpha$={int(alpha)}  $\Delta F/N$={dF_ti:.4f}")
    ax_intg.fill_between(lam_np, 0, mean_diff, color=c, alpha=0.18)

    # Cumulative FE path
    ax_cfe.plot(lam_np, fpath, color=c, lw=1.8,
                label=rf"$\alpha$={int(alpha)}: {fpath[-1]:.4f}")

    # MSD with equipartition reference: <MSD> = 3T/α  per atom
    equi_msd = 3.0 * T_sim / float(alpha)
    ax_msd.plot(lam_np, mean_msd, color=c, lw=1.8,
                label=rf"$\alpha$={int(alpha)}")
    ax_msd.axhline(equi_msd, color=c, lw=1.2, ls=":",
                   label=rf"equip. $3T/\alpha={equi_msd:.4f}$")

ax_intg.axhline(0, color="k", lw=0.7, ls="--")
ax_cfe.axhline(eos_aabs, color="k", ls="--", lw=1.6, label=f"EOS {eos_aabs:.4f}")

ax_intg.set_xlabel(r"$\lambda$")
ax_intg.set_ylabel(r"$\langle U_{LJ}-U_{EC}\rangle_\lambda$")
ax_intg.set_title("TI integrand  (shaded area = ΔF)", fontsize=10)
ax_intg.legend(fontsize=8)

ax_cfe.set_xlabel(r"$\lambda$")
ax_cfe.set_ylabel(r"$a_\mathrm{abs}/N$")
ax_cfe.set_title("TI cumulative free energy path", fontsize=10)
ax_cfe.legend(fontsize=8)

ax_msd.set_xlabel(r"$\lambda$")
ax_msd.set_ylabel("MSD")
ax_msd.set_title(r"MSD vs $\lambda$ with equipartition reference ($3T/\alpha$, dotted)", fontsize=10)
ax_msd.legend(fontsize=8, ncol=3)

fig_ti.suptitle("Thermodynamic Integration  |  " + META, fontsize=12, y=1.01)
fig_ti.tight_layout()
out = os.path.join(FIGS, "ti_plot.png")
fig_ti.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
plt.close(fig_ti)

# =====================================================================
# 2. Bar comparison
# =====================================================================
def ti_an(alpha):
    return float(ti[f"a{int(alpha)}_F_path_per_atom_abs"][-1])

def outer_cr_an(alpha):
    return float(outer_cr[f"a{int(alpha)}_a_abs_est"][0])

def ca_an(alpha, sched):
    r = next(r for r in cr_runs if r["alpha"] == alpha and r["schedule"] == sched)
    return r["a_abs_est"]

methods = ["TI", "linear sw2000 (orig)", "cosine sw2000", "linear sw2000 (new)"]
all_vals = [
    [ti_an(a)              for a in alphas],
    [outer_cr_an(a)        for a in alphas],
    [ca_an(a, "cosine")    for a in alphas],
    [ca_an(a, "linear")    for a in alphas],
]
bar_colors = ["#555555", "#888888",
              plt.cm.tab10.colors[0], plt.cm.tab10.colors[1]]
hatches    = ["", "xx", "", "//"]

fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
x = np.arange(len(alphas))
w = 0.8 / len(methods)

for mi, (vals, lbl, col, hatch) in enumerate(zip(all_vals, methods, bar_colors, hatches)):
    offset = (mi - len(methods) / 2 + 0.5) * w
    ax_bar.bar(x + offset, vals, width=w * 0.9,
               label=lbl, color=col, hatch=hatch,
               edgecolor="k", linewidth=0.6, alpha=0.85)

ax_bar.axhline(eos_aabs, color="k", ls="--", lw=1.8, label=f"EOS {eos_aabs:.4f}")
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([rf"$\alpha$={int(a)}" for a in alphas])
ax_bar.set_ylabel(r"$a_\mathrm{abs}/N$", fontsize=12)
ax_bar.set_title(r"$a/N$ estimate by method  |  " + META, fontsize=10)
ax_bar.legend(fontsize=9)

all_flat = [v for row in all_vals for v in row] + [eos_aabs]
pad = 0.1 * (max(all_flat) - min(all_flat) + 1e-9)
ax_bar.set_ylim(min(all_flat) - pad, max(all_flat) + pad)

fig_bar.tight_layout()
out = os.path.join(FIGS, "bar_comparison.png")
fig_bar.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
plt.close(fig_bar)

# =====================================================================
# 3a. Original Crooks plots (from parent crooks_results.npz)
# =====================================================================
region_patches = [
    Patch(color="gold",        alpha=0.4, label="eq λ=0"),
    Patch(color="lightgreen",  alpha=0.4, label="fwd switch"),
    Patch(color="lightsalmon", alpha=0.4, label="eq λ=1"),
    Patch(color="plum",        alpha=0.4, label="rev switch"),
]

outer_sw = int(outer_cr["steps_switch"][0])
outer_eq = 1000   # original steps_eq not saved; matches original script

for alpha in alphas:
    pfx = f"a{int(alpha)}_"
    c   = alpha_colors[alpha]

    W_fwd = np.asarray(outer_cr[pfx + "W_fwd"])
    W_rev = np.asarray(outer_cr[pfx + "W_rev"])
    w_grid   = np.asarray(outer_cr[pfx + "w_grid"])
    p_f      = np.asarray(outer_cr[pfx + "p_f"])
    p_r      = np.asarray(outer_cr[pfx + "p_r"])
    dF_crooks = float(outer_cr[pfx + "dF_crooks"][0])
    dF_target = float(outer_cr[pfx + "dF_target"][0])
    a_abs_est = float(outer_cr[pfx + "a_abs_est"][0])

    fig_cr, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"wspace": 0.32})

    Wf = W_fwd
    Wr = -W_rev
    wlo, whi = min(Wf.min(), Wr.min()), max(Wf.max(), Wr.max())
    bins = np.linspace(wlo, whi, 50)
    ax_h.hist(Wf, bins=bins, density=True, alpha=0.22, color=c)
    ax_h.hist(Wr, bins=bins, density=True, alpha=0.22, color="gray")
    ax_h.plot(w_grid, p_f, color=c,   lw=2.0, label=r"$P_F(W)$")
    ax_h.plot(w_grid, p_r, color="k", lw=1.4, ls="--", label=r"$P_R(-W)$")
    overlap = np.minimum(p_f, p_r)
    ax_h.fill_between(w_grid, 0, overlap, color="green", alpha=0.15, label="overlap")
    ax_h.axvline(dF_crooks, color="red",  lw=1.8,
                 label=rf"Crooks $\Delta F$ = {dF_crooks:.3f}")
    ax_h.axvline(dF_target, color="gray", lw=1.4, ls=":",
                 label=rf"EOS target = {dF_target:.3f}")
    ax_h.set_xlabel("Work $W$", fontsize=10)
    ax_h.set_ylabel("Probability density", fontsize=10)
    ax_h.legend(fontsize=8)
    ax_h.set_title(
        rf"$\alpha$={int(alpha)}, linear sw={outer_sw}, eq={outer_eq}"
        "\n"
        rf"$\hat{{a}}/N$ = {a_abs_est:.5f}    EOS = {eos_aabs:.5f}    "
        rf"err = {a_abs_est - eos_aabs:+.5f}",
        fontsize=9,
    )

    # linear schedule (original used linspace(1e-3, 1.0))
    t_sw = np.linspace(1e-3, 1.0, outer_sw + 1)
    lam_fwd  = t_sw
    lam_rev  = lam_fwd[::-1]
    lam_full = np.concatenate([np.zeros(outer_eq), lam_fwd,
                               np.ones(outer_eq), lam_rev])
    t_full   = np.arange(len(lam_full))

    ax_s.plot(t_full, lam_full, color="purple", lw=2.0)
    t0, t1 = 0, outer_eq
    t2 = t1 + outer_sw + 1
    t3 = t2 + outer_eq
    ax_s.axvspan(t0, t1, alpha=0.12, color="gold")
    ax_s.axvspan(t1, t2, alpha=0.12, color="lightgreen")
    ax_s.axvspan(t2, t3, alpha=0.12, color="lightsalmon")
    ax_s.axvspan(t3, len(lam_full), alpha=0.12, color="plum")
    ax_s.set_ylim(-0.05, 1.05)
    ax_s.set_ylabel(r"$\lambda$", fontsize=11)
    ax_s.set_xlabel("MD step", fontsize=10)
    ax_s.set_title("Lambda schedule", fontsize=10)
    ax_s.legend(handles=region_patches, fontsize=8, ncol=2, loc="upper center")

    fname = f"crooks_orig_alpha{int(alpha)}.png"
    out = os.path.join(FIGS, fname)
    fig_cr.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig_cr)

# =====================================================================
# 3b. Per-run Crooks plots (histogram top, schedule bottom)
# =====================================================================
for r in cr_runs:
    alpha = r["alpha"]
    sched = r["schedule"]
    c     = alpha_colors[alpha]
    sw, eq = r["steps_switch"], r["steps_eq"]

    fig_cr, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"wspace": 0.32})

    # -- histogram
    Wf = r["W_fwd"]
    Wr = -r["W_rev"]
    wlo, whi = min(Wf.min(), Wr.min()), max(Wf.max(), Wr.max())
    bins = np.linspace(wlo, whi, 50)
    ax_h.hist(Wf, bins=bins, density=True, alpha=0.22, color=c)
    ax_h.hist(Wr, bins=bins, density=True, alpha=0.22, color="gray")
    ax_h.plot(r["w_grid"], r["p_f"], color=c,   lw=2.0, label=r"$P_F(W)$")
    ax_h.plot(r["w_grid"], r["p_r"], color="k", lw=1.4, ls="--",
              label=r"$P_R(-W)$")
    overlap = np.minimum(r["p_f"], r["p_r"])
    ax_h.fill_between(r["w_grid"], 0, overlap, color="green", alpha=0.15,
                      label="overlap")
    ax_h.axvline(r["dF_crooks"], color="red",  lw=1.8,
                 label=rf"Crooks $\Delta F$ = {r['dF_crooks']:.3f}")
    ax_h.axvline(r["dF_target"], color="gray", lw=1.4, ls=":",
                 label=rf"EOS target = {r['dF_target']:.3f}")
    ax_h.set_xlabel("Work $W$", fontsize=10)
    ax_h.set_ylabel("Probability density", fontsize=10)
    ax_h.legend(fontsize=8)
    ax_h.set_title(
        rf"$\alpha$={int(alpha)}, schedule={sched}, sw={sw}, eq={eq}"
        "\n"
        rf"$\hat{{a}}/N$ = {r['a_abs_est']:.5f}    EOS = {r['eos_aabs']:.5f}    "
        rf"err = {r['a_abs_est'] - r['eos_aabs']:+.5f}",
        fontsize=9,
    )

    # -- schedule
    t_sw    = np.linspace(0.0, 1.0, sw + 1)
    lam_fwd = 0.5 * (1.0 - np.cos(np.pi * t_sw)) if sched == "cosine" else t_sw
    lam_rev = lam_fwd[::-1]
    lam_full = np.concatenate([np.zeros(eq), lam_fwd, np.ones(eq), lam_rev])
    t_full   = np.arange(len(lam_full))

    ax_s.plot(t_full, lam_full, color=sched_col[sched],
              lw=2.0, ls=sched_ls[sched])
    t0, t1, t2, t3 = 0, eq, eq + sw + 1, eq + sw + 1 + eq
    ax_s.axvspan(t0, t1, alpha=0.12, color="gold")
    ax_s.axvspan(t1, t2, alpha=0.12, color="lightgreen")
    ax_s.axvspan(t2, t3, alpha=0.12, color="lightsalmon")
    ax_s.axvspan(t3, len(lam_full), alpha=0.12, color="plum")
    ax_s.set_ylim(-0.05, 1.05)
    ax_s.set_ylabel(r"$\lambda$", fontsize=11)
    ax_s.set_xlabel("MD step", fontsize=10)
    ax_s.set_title("Lambda schedule", fontsize=10)
    ax_s.legend(handles=region_patches, fontsize=8, ncol=2,
                loc="upper center")

    fname = f"crooks_alpha{int(alpha)}_{sched}.png"
    out = os.path.join(FIGS, fname)
    fig_cr.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig_cr)

# =====================================================================
# 4. Parameter tables
# =====================================================================
a_lat  = 1.54174
L_box  = a_lat * 7
rho_st = float(rho_star)

HDR  = "#2c4a7c"   # dark navy header fill
ROW0 = "#f0f4ff"   # light blue-tint for even rows
ROW1 = "#ffffff"   # white for odd rows
TXT  = "white"

def _styled_table(ax, col_labels, row_data, col_widths=None):
    """Draw a header-styled table on `ax`, returns the Table object."""
    ax.axis("off")
    n_cols = len(col_labels)
    n_rows = len(row_data)

    tbl = ax.table(
        cellText=row_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    if col_widths:
        for ci, w in enumerate(col_widths):
            for ri in range(-1, n_rows):   # -1 is the header row
                tbl[ri + 1, ci].set_width(w)

    # Header row styling
    for ci in range(n_cols):
        cell = tbl[0, ci]
        cell.set_facecolor(HDR)
        cell.set_text_props(color=TXT, fontweight="bold", fontsize=10)
        cell.set_edgecolor("#aaaaaa")
        cell.set_height(0.13)

    # Data rows
    for ri in range(n_rows):
        bg = ROW0 if ri % 2 == 0 else ROW1
        for ci in range(n_cols):
            cell = tbl[ri + 1, ci]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#cccccc")
            cell.set_height(0.10)
            # left-align first column
            if ci == 0:
                cell.set_text_props(ha="left", fontweight="semibold")

    tbl.scale(1, 1)
    return tbl


# --- Table 1: System parameters ---
sys_cols = ["Parameter", "Value"]
sys_rows = [
    ["Lattice",              "FCC, 7×7×7 unit cells"],
    ["Lattice constant a",   f"{a_lat}  (LJ units)"],
    ["Box length L",         f"{L_box:.4f}  (LJ units)"],
    ["N atoms",              f"{N_atoms}"],
    ["ρ*  =  N / L³",        f"{rho_st:.6f}"],
    ["Temperature T",        "1.0  (LJ reduced)"],
    ["Batch size B",         "256 simultaneous trajectories"],
    ["Integrator",           "Langevin NVT"],
    ["Timestep dt",          "0.005"],
    ["Friction γ",           "10.0"],
    ["LJ potential",         "Mie  n=12,  σ=1.0,  ε=1.0"],
    ["Reference EOS",        "van der Hoef FCC-LJ"],
    ["Spring constants α",   "150,  200,  250"],
    ["EOS  a/N  (target)",   f"{eos_aabs:.6f}"],
]

# --- Table 2: Method comparison ---
mth_cols = ["", "TI", "Crooks orig", "CrooksAgain cosine", "CrooksAgain linear"]
mth_rows = [
    ["Script",           "einstein_nvt.py", "einstein_nvt_jarzynski.py", "einstein_crooks.py", "einstein_crooks.py"],
    ["λ schedule",       "256-pt linspace\n1e-5 → 1", "linear\n1e-3 → 1", "cosine\n0 → 1", "linear\n0 → 1"],
    ["steps_eq",         "1 000", "1 000", "2 000", "2 000"],
    ["steps_switch",     "—", "2 000", "2 000", "3 000"],
    ["steps_prod",       "5 000 / λ", "—", "—", "—"],
    ["n_passes",         "—", "10", "10", "10"],
    ["Total traj / dir", "256", "2 560", "2 560", "2 560"],
    ["dW estimator",     "⟨U_LJ−U_EC⟩dλ", "dλ·(E_LJ−E_EC)", "U(λ+dλ)−U(λ)", "U(λ+dλ)−U(λ)"],
    ["ΔF estimator",     "Trapezoid", "Crooks KDE", "Crooks KDE", "Crooks KDE"],
    ["COM constraint",   "mom only", "pos + mom", "pos + mom", "pos + mom"],
]

fig_tbl, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(13, 11),
    gridspec_kw={"hspace": 0.18, "top": 0.96, "bottom": 0.02,
                 "left": 0.02, "right": 0.98}
)

_styled_table(ax1, sys_cols,  sys_rows,  col_widths=[0.38, 0.62])
_styled_table(ax2, mth_cols,  mth_rows,  col_widths=[0.20, 0.17, 0.21, 0.21, 0.21])

ax1.set_title("System Parameters", fontsize=13, fontweight="bold",
              pad=6, loc="left", color=HDR)
ax2.set_title("Method Comparison", fontsize=13, fontweight="bold",
              pad=6, loc="left", color=HDR)

out = os.path.join(FIGS, "params_table.png")
fig_tbl.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
plt.close(fig_tbl)
