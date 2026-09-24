"""Demonstration: classification and density-ratio estimation are different TASKS.

Left: two DIFFERENT processes (HH signal vs ttbar-like background) in a feature plane -
classification learns one decision boundary; only which side an event falls on (the
ordering) matters, and the score's calibration is irrelevant for ranking.

Right: the SAME di-Higgs process at two kappa_lambda values - the two densities occupy
the same phase space (nothing to separate, AUC ~ 0.7 at best); the deliverable is the
VALUE of r(x) = p(x|kl')/p(x|kl) at every x, a calibrated function over the full
support that enters the likelihood event by event.

  python docs/teaching/boundary_vs_ratio_toy.py   -> boundary_vs_ratio_toy.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)

# ---- left: two processes in a 2-feature plane + the optimal boundary ----
MU_S, COV_S = [118.0, 1.55], [[620.0, -2.0], [-2.0, 0.22]]   # equal covariances ->
MU_B, COV_B = [172.0, 0.70], [[620.0, -2.0], [-2.0, 0.22]]   # the boundary is one line
xs = rng.multivariate_normal(MU_S, COV_S, 350)
xb = rng.multivariate_normal(MU_B, COV_B, 350)


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


fig = plt.figure(figsize=(12.5, 4.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1.6, 1],
                      hspace=0.08, wspace=0.25)
axA = fig.add_subplot(gs[:, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, 1], sharex=axB)

axA.scatter(*xs.T, s=9, alpha=0.5, color="#2F6BDE", label=r"$HH$ signal", lw=0)
axA.scatter(*xb.T, s=9, alpha=0.5, color="#CC6611", label=r"$t\bar t$-like background", lw=0)
g1, g2 = np.meshgrid(np.linspace(40, 260, 300), np.linspace(-0.4, 2.9, 300))
G = np.stack([g1, g2], axis=-1)
axA.contour(g1, g2, _gauss2(G, MU_S, COV_S) / _gauss2(G, MU_B, COV_B), levels=[1.0],
            colors="0.15", linestyles="--", linewidths=1.8)
axA.annotate("decision boundary\n$p(S\\,|\\,x)=0.5$", xy=(163, 1.75), xytext=(205, 2.35),
             fontsize=9, arrowprops=dict(arrowstyle="-", color="0.4", lw=0.9))
axA.text(0.03, 0.03, "one number learned per region of space:\nwhich side? "
         "(ordering only -- calibration never enters)", transform=axA.transAxes, fontsize=8.5,
         color="0.3")
axA.set_xlabel(r"$m_{bb}$-like feature"); axA.set_ylabel(r"$\tau\tau$-kinematic feature")
axA.set_title("classification: two DIFFERENT processes\n= learn a boundary between groups",
              fontsize=10)
axA.legend(fontsize=8.5, frameon=False, loc="upper left")

p1, p5 = lineshape(1.0), lineshape(5.0)
axB.fill_between(X, p1, color="0.45", alpha=0.25)
axB.plot(X, p1, color="0.35", lw=2.0, ls="--", label=r"$\kappa_\lambda=1$")
axB.fill_between(X, p5, color="#009988", alpha=0.18)
axB.plot(X, p5, color="#009988", lw=2.0, label=r"$\kappa_\lambda=5$")
axB.text(0.97, 0.55, "same process, same phase space:\nnothing to separate --\n"
         "every event is 'both'", transform=axB.transAxes, fontsize=8.5, color="0.3",
         ha="right")
axB.set_ylabel(r"$p(x\,|\,\kappa_\lambda)$")
axB.set_title("density-ratio estimation: the SAME di-Higgs process\nat two "
              r"$\kappa_\lambda$ = learn a function over all of $x$", fontsize=10)
axB.legend(fontsize=8.5, frameon=False, loc="upper right")
plt.setp(axB.get_xticklabels(), visible=False)

axC.plot(X, p5 / p1, color="#009988", lw=2.0)
axC.axhline(1.0, color="0.25", lw=1, ls="--")
axC.set_yscale("log"); axC.set_ylim(1e-3, 30)
axC.set_xlabel(r"$m_{HH}$-like observable $x$")
axC.set_ylabel(r"$r(x)=\frac{p(x|5)}{p(x|1)}$")
axC.text(0.03, 0.78, "the deliverable: the VALUE of $r(x)$ everywhere\n"
         "(enters the likelihood event by event)", transform=axC.transAxes, fontsize=8.5,
         color="0.3")

for ax in (axA, axB, axC):
    ax.grid(True, alpha=0.18, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Two different learning tasks:\nseparating processes (a boundary, judged by "
             "ordering)  vs  comparing hypotheses within one process (a function, judged "
             "by its value)", fontsize=10.5)
fig.subplots_adjust(top=0.80, bottom=0.11, left=0.07, right=0.985)
out = __file__.replace(".py", ".png")
fig.savefig(out, dpi=160)
print("wrote", out)
