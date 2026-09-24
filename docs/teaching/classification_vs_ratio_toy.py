"""Demonstration: classification difficulty and ratio-estimation difficulty INVERT with lambda.

Companion to learning-records/0011-classification-vs-ratio-estimation-difficulty.md.
Toy m_HH-like lineshapes p(x|kl): a threshold bump whose weight grows with kl (the
triangle-box interference caricature) over a broad tail. For each kl the OPTIMAL
classifier against the kl=1 reference is the likelihood ratio itself, so the ROC/AUC
shown is the ceiling any trained classifier can reach - no training noise involved.

  python docs/teaching/classification_vs_ratio_toy.py   -> classification_vs_ratio_toy.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

X = np.linspace(250.0, 1000.0, 4000)
# threshold-bump weight AND tail shape morph with kl: for kl=5 the spectrum concentrates
# at threshold and the hard tail dies, so the ratio to kl=1 loses support at high mass
PARS = {0.0: (0.15, 560.0, 150.0), 1.0: (0.35, 540.0, 140.0),
        2.0: (0.42, 535.0, 138.0), 5.0: (0.88, 470.0, 80.0)}   # (bump weight, tail mu, tail sigma)
COL = {0.0: "#CC6611", 1.0: "0.25", 2.0: "#009988", 5.0: "#2F6BDE"}


def density(kl):
    w, mu_t, sig_t = PARS[kl]
    bump = np.exp(-0.5 * ((X - 380.0) / 45.0) ** 2)
    tail = np.exp(-0.5 * ((X - mu_t) / sig_t) ** 2)
    p = w * bump / bump.sum() + (1 - w) * tail / tail.sum()
    return p / np.trapz(p, X)


def roc(p_num, p_ref):
    order = np.argsort(-(p_num / p_ref))                  # optimal classifier = the ratio
    w_n, w_r = (p_num[order] / p_num.sum()), (p_ref[order] / p_ref.sum())
    tpr, fpr = np.concatenate([[0], np.cumsum(w_n)]), np.concatenate([[0], np.cumsum(w_r)])
    return fpr, tpr, float(np.trapz(tpr, fpr))


p_ref = density(1.0)
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.5, 4.2))

for kl in (0.0, 2.0, 5.0, 1.0):
    a1.plot(X, density(kl), color=COL[kl], lw=2.2 if kl == 1.0 else 1.8,
            ls="-" if kl != 1.0 else "--",
            label=rf"$\kappa_\lambda={kl:g}$" + (" (reference)" if kl == 1.0 else ""))
a1.set_xlabel(r"$m_{HH}$-like observable $x$"); a1.set_ylabel(r"$p(x\,|\,\kappa_\lambda)$")
a1.set_title("toy lineshapes", fontsize=10)
a1.legend(fontsize=8.5, frameon=False)

for kl in (0.0, 2.0, 5.0):
    fpr, tpr, auc = roc(density(kl), p_ref)
    a2.plot(fpr, tpr, color=COL[kl], lw=1.8,
            label=rf"$\kappa_\lambda={kl:g}$ vs 1:  AUC = {auc:.3f}")
a2.plot([0, 1], [0, 1], color="0.7", lw=1, ls=":")
a2.set_xlabel("false positive rate"); a2.set_ylabel("true positive rate")
a2.set_title("classification vs the reference:\nseparation is fuel "
             r"$\rightarrow$ distant $\kappa_\lambda$ is EASY", fontsize=10)
a2.legend(fontsize=8.5, frameon=False, loc="lower right")

for kl in (0.0, 2.0, 5.0):
    r = density(kl) / p_ref
    lo = f"{r.min():.1f}" if r.min() > 0.05 else f"$10^{{{np.log10(r.min()):.0f}}}$"
    a3.plot(X, r, color=COL[kl], lw=1.8,
            label=rf"$\kappa_\lambda={kl:g}$:  range {lo}--{r.max():.1f}")
a3.axhline(1.0, color="0.25", lw=1, ls="--")
a3.set_yscale("log"); a3.set_ylim(1e-4, 40)
a3.set_xlabel(r"$m_{HH}$-like observable $x$")
a3.set_ylabel(r"$r(x) = p(x|\kappa_\lambda)\,/\,p(x|1)$")
a3.set_title("the ratio the likelihood consumes:\ndynamic range is the enemy "
             r"$\rightarrow$ distant $\kappa_\lambda$ is HARD", fontsize=10)
a3.legend(fontsize=8.5, frameon=False, loc="lower left")

for ax in (a1, a2, a3):
    ax.grid(True, alpha=0.18, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("The two difficulties invert: pairs easiest to CLASSIFY (high AUC) have the "
             "hardest RATIOS (unbounded target where reference support dies) - and vice versa.\n"
             "AUC-style evidence therefore cannot certify a density ratio; the near-unity "
             "regime (close hypotheses, systematics reweighting) is where ratios are benign.",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.88))
out = __file__.replace(".py", ".png")
fig.savefig(out, dpi=160)
print("wrote", out)
