"""Demonstration: classification and density-ratio estimation are different TASKS.

Left: two DIFFERENT processes (HH signal vs ttbar-like background) in a feature plane -
classification learns one decision boundary; only which side an event falls on (the
ordering) matters, and the score's calibration is irrelevant for ranking.

Right: the SAME di-Higgs process at two kappa_lambda values - the two densities occupy
the same phase space (nothing to separate); the deliverable is the VALUE of
r(x) = p(x|kl')/p(x|kl) at every x, a calibrated function over the full support that
enters the likelihood event by event.

Drawn in the teaching-schematic style (cream ground, serif, brick-red curves with a
soft band, curved annotations), matching the sigma-vs-kl schematic in the lessons.

  python docs/teaching/boundary_vs_ratio_toy.py   -> boundary_vs_ratio_toy.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CREAM, INK, SOFT = "#FAF7F1", "#3A3430", "#6E645C"
RED, REDBAND = "#8B1F1F", "#8B1F1F"
GRAY, TAN = "#55504B", "#B0977B"
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "figure.facecolor": CREAM, "axes.facecolor": CREAM, "savefig.facecolor": CREAM,
    "text.color": INK, "axes.edgecolor": SOFT, "axes.labelcolor": INK,
    "xtick.color": SOFT, "ytick.color": SOFT,
    "axes.spines.top": False, "axes.spines.right": False,
})


def banded(ax, x, y, color=RED, lw=2.2, band=11, **kw):
    """The signature soft band: the same line drawn wide and faint underneath."""
    ax.plot(x, y, color=color, lw=band, alpha=0.14, solid_capstyle="round", zorder=1)
    ax.plot(x, y, color=color, lw=lw, zorder=2, **kw)


rng = np.random.default_rng(11)

# ---- left: two processes in a 2-feature plane + the optimal boundary ----
MU_S, COV_S = [118.0, 1.55], [[620.0, -2.0], [-2.0, 0.22]]   # equal covariances ->
MU_B, COV_B = [172.0, 0.70], [[620.0, -2.0], [-2.0, 0.22]]   # the boundary is one line
xs = rng.multivariate_normal(MU_S, COV_S, 320)
xb = rng.multivariate_normal(MU_B, COV_B, 320)


def _gauss2(g, mu, cov):
    d = g - np.asarray(mu)
    ic = np.linalg.inv(cov)
    return np.exp(-0.5 * np.einsum("...i,ij,...j->...", d, ic, d)) / np.sqrt(np.linalg.det(cov))


# ---- right: one process, two kappa_lambda lineshapes + the ratio function ----
X = np.linspace(250.0, 1000.0, 2000)
PARS = {1.0: (0.35, 540.0, 140.0), 5.0: (0.88, 470.0, 80.0)}   # (bump weight, tail mu, sigma)


def lineshape(kl):
    w, mu_t, sig_t = PARS[kl]
    bump = np.exp(-0.5 * ((X - 380.0) / 45.0) ** 2)
    tail = np.exp(-0.5 * ((X - mu_t) / sig_t) ** 2)
    p = w * bump / bump.sum() + (1 - w) * tail / tail.sum()
    return p / np.trapezoid(p, X)


fig = plt.figure(figsize=(13, 5.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1.55, 1],
                      hspace=0.10, wspace=0.24)
axA = fig.add_subplot(gs[:, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 1], sharex=axB)

axA.scatter(*xs.T, s=11, alpha=0.55, color=RED, lw=0, label=r"$HH$ signal")
axA.scatter(*xb.T, s=11, alpha=0.55, color=TAN, lw=0, label=r"$t\bar t$-like background")
g1, g2 = np.meshgrid(np.linspace(45, 255, 250), np.linspace(-1.1, 3.3, 250))
G = np.stack([g1, g2], axis=-1)
bnd = axA.contour(g1, g2, _gauss2(G, MU_S, COV_S) / _gauss2(G, MU_B, COV_B), levels=[1.0],
                  colors=INK, linestyles="--", linewidths=2.0)
seg = max(bnd.allsegs[0], key=len)
axA.plot(seg[:, 0], seg[:, 1], color=INK, lw=11, alpha=0.10, solid_capstyle="round", zorder=0)
mid = seg[int(0.30 * len(seg))]
axA.annotate("the whole learned object:\none boundary \u2014 which side?\n"
             "(ordering only \u2014 calibration\nnever enters)",
             xy=(mid[0] + 3, mid[1]), xytext=(0.55, 0.02), textcoords="axes fraction",
             fontsize=10.5, color=SOFT, ha="left",
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.3,
                             connectionstyle="arc3,rad=0.3"))
axA.set_xlabel(r"$m_{bb}$-like feature", fontsize=11)
axA.set_ylabel(r"$\tau\tau$-kinematic feature", fontsize=11)
axA.set_title("classification: two different processes\n= learn a boundary between groups",
              fontsize=12)
axA.legend(fontsize=10, frameon=False, loc="upper left", labelcolor=INK)
axA.set_xticks([50, 100, 150, 200, 250]); axA.set_yticks([0, 1, 2, 3])

p1, p5 = lineshape(1.0), lineshape(5.0)
banded(axB, X, p1, color=GRAY, lw=2.0, band=9, ls="--", label=r"$\kappa_\lambda=1$")
banded(axB, X, p5, label=r"$\kappa_\lambda=5$")
axB.text(0.97, 0.52, "same process, same phase space:\nnothing to separate",
         transform=axB.transAxes, fontsize=10.5, color=SOFT, ha="right")
axB.set_ylabel(r"$p(x\,|\,\kappa_\lambda)$", fontsize=11)
axB.set_title("density-ratio estimation: the same di-Higgs\nprocess at two "
              r"$\kappa_\lambda$ = learn a function of $x$", fontsize=12)
axB.legend(fontsize=10, frameon=False, loc="upper right", labelcolor=INK)
axB.set_yticks([]); axB.set_ylim(0, None)
plt.setp(axB.get_xticklabels(), visible=False)

r = p5 / p1
banded(axC, X, r)
axC.axhline(1.0, color=SOFT, lw=1.0, ls=":")
axC.set_yscale("log"); axC.set_ylim(1e-3, 60)
axC.set_yticks([1e-2, 1e0])
axC.set_xlabel(r"$m_{HH}$-like observable $x$", fontsize=11)
axC.set_ylabel(r"$r(x)$", fontsize=11)
axC.annotate("the deliverable: the value of\n$r(x)$ at every $x$ \u2014 it enters\n"
             "the likelihood event by event", xy=(640, r[np.searchsorted(X, 640)]),
             xytext=(0.02, 0.10), textcoords="axes fraction", fontsize=10.5, color=SOFT,
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.3,
                             connectionstyle="arc3,rad=0.3"))
axC.set_xticks([300, 500, 700, 900])

fig.suptitle("classification vs density-ratio estimation \u2014 schematic "
             "(a boundary, judged by ordering  vs  a function, judged by its value)",
             fontsize=13.5, fontweight="bold", y=0.97)
fig.subplots_adjust(top=0.82, bottom=0.11, left=0.06, right=0.985)
out = __file__.replace(".py", ".png")
fig.savefig(out, dpi=160)
print("wrote", out)
