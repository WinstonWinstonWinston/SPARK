import os
import numpy as np
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.abspath(__file__))
crk = np.load(os.path.join(dir_path, "crooks_results.npz"), allow_pickle=False)

alpha_values = crk["alpha_values"].tolist()
ncols = len(alpha_values)

fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), squeeze=False)
axes = axes[0]

colors = plt.cm.tab10.colors

for idx, alpha in enumerate(alpha_values):
    pfx = f"a{int(alpha)}_"
    ax  = axes[idx]
    c   = colors[idx % len(colors)]

    W_fwd = crk[pfx + "W_fwd"]
    W_rev = crk[pfx + "W_rev"]

    ax.hist(W_fwd,  bins=40, density=True, alpha=0.5, color=c,       label=r"$W_\mathrm{fwd}$")
    ax.hist(-W_rev, bins=40, density=True, alpha=0.5, color=c,
            histtype="step", linewidth=1.5,                            label=r"$-W_\mathrm{rev}$")

    ax.set_title(rf"$\alpha$ = {alpha:.0f}")
    ax.set_xlabel("Work")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

fig.tight_layout()
out = os.path.join(dir_path, "work_histograms.png")
fig.savefig(out, dpi=150)
print(f"Saved {out}")
