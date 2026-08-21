"""Figures for EVENET_G0_RESULTS.tex, regenerated from the committed result JSONs.

  python scripts/plot_note_figures.py [--ens5 results/ensemble-k5.json]
         [--ens20 results/ensemble-k20.json] [--closure20 results/closure-k20.json]
         [--closure5 <closure-trained.json>] [--outdir figures] [--png]

Writes vector PDFs (the repo LFS-tracks *.png, so figures are committed as PDF):
  auc_efficiency.pdf      Result 1: ensemble AUC vs fraction, both campaigns + BDT ceiling
  closure_trends.pdf      Result 2: |IC| and SC_rms vs fraction, K=5 and K=20 rows separate
  noise_decomposition.pdf Sec. power: seed term vs test floor on the ensemble |IC|, K=5 vs K=20
  p5_verdict.pdf          Result 3: pre-registered Delta per fraction + averages, and the
                          AUC advantage on the same ensembles

Error bars. K=20 cells: paired-bootstrap test error (+) per_seed_ic_std/sqrt(K), K the
cell's actual member count; SC seed term = single-model seed spread from --closure20 / sqrt(K)
(SC is nonlinear in the members, so this is a proxy). K=5 cells: Gaussian stat error (+)
per_seed_ic_std/sqrt(5); SC seed term from --closure5 if given, else from the K=5 sweep's
single-model spreads embedded below (transcribed from closure-trained.json, Perlmutter).
"""
import argparse
import json
import os

import numpy as np

COL = {"finetune": "#2F6BDE", "frozen": "#009988", "scratch": "#CC6611"}   # CVD-validated
FR5 = ["0.003", "0.01", "0.02", "0.05", "0.1", "1.0"]
FR20 = ["0.003", "0.01", "0.02", "0.05", "0.1"]
CEILING = {"0.01": 0.7917, "0.1": 0.8130, "1.0": 0.8174}   # tab:ceilingtab (BDT, 10 features)
SC5_PROXY = {   # single-model SC_rms seed s.d., seeds 0-4 (closure-trained.json, 2026-07-17)
    "finetune": {"0.003": .0851, "0.01": .1179, "0.02": .0908, "0.05": .0298, "0.1": .0342, "1.0": .0376},
    "frozen":   {"0.003": .1501, "0.01": .2112, "0.02": .0862, "0.05": .0442, "0.1": .0738, "1.0": .0483},
    "scratch":  {"0.003": .1869, "0.01": .1163, "0.02": .0586, "0.05": .0451, "0.1": .0579, "1.0": .0562}}


def _x(fracs, k):
    return np.array([float(f) for f in fracs]) * (0.92 + 0.08 * k)


def _style(ax, ylab=None, xlab=True, log_y=True):
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    if xlab:
        ax.set_xlabel("training fraction")
    if ylab:
        ax.set_ylabel(ylab)
    ax.grid(True, which="both", alpha=0.18, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def fig_auc(E5, E20, out):
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for k, a in enumerate(COL):
        a1.plot(_x(FR5, k), [E5[a][f]["auc"] for f in FR5], marker="o", ms=6, mfc="white",
                mew=1.6, lw=1.2, ls="--", color=COL[a], label=f"{a} (K=5)")
        a1.errorbar(_x(FR20, k), [E20[a][f]["auc"] for f in FR20],
                    yerr=[E20[a][f]["boot_auc_std"] for f in FR20], marker="o", ms=5,
                    lw=1.8, capsize=2, color=COL[a], label=f"{a} (K=20)")
    a1.plot([float(f) for f in CEILING], list(CEILING.values()), marker="x", ms=8, mew=2,
            ls=":", color="0.35", label="BDT ceiling (10 features)")
    _style(a1, "ensemble AUC", log_y=False)
    a1.legend(fontsize=7.5, frameon=False, ncol=2, loc="lower right")
    a1.set_title("ensemble AUC, both campaigns", fontsize=10)
    for a, dx in {"finetune": 0.92, "frozen": 1.08}.items():
        d5 = [E5[a][f]["auc"] - E5["scratch"][f]["auc"] for f in FR5]
        a2.plot(np.array([float(f) for f in FR5]) * dx, d5, marker="o", ms=6, mfc="white",
                mew=1.6, lw=1.2, ls="--", color=COL[a], label=f"{a} (K=5)")
        d20 = [E20[a][f]["auc"] - E20["scratch"][f]["auc"] for f in FR20]
        e20 = [np.hypot(E20[a][f]["boot_auc_std"], E20["scratch"][f]["boot_auc_std"])
               for f in FR20]
        a2.errorbar(np.array([float(f) for f in FR20]) * dx, d20, yerr=e20, marker="o",
                    ms=5, lw=1.8, capsize=2, color=COL[a], label=f"{a} (K=20)")
    a2.axhline(0, color="0.25", lw=1)
    _style(a2, r"$\mathrm{AUC}_{\rm arm}-\mathrm{AUC}_{\rm scratch}$", log_y=False)
    a2.legend(fontsize=7.5, frameon=False, loc="lower right")
    a2.set_title("advantage over scratch (up = pre-training better)", fontsize=10)
    fig.suptitle("Discrimination: pre-training wins at every fraction $\\geq 0.01$ "
                 "(K=20 bars: test-set bootstrap)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)


def fig_closure(E5, E20, C20, C5, out):
    import matplotlib.pyplot as plt

    def ic5(a, f):
        m = E5[a][f]
        return abs(m["integral"]), np.hypot(m["integral_err"], m["per_seed_ic_std"] / np.sqrt(5))

    def sc5(a, f):
        sd = (np.std([x["shape_rms"] for x in C5[a][f]]) if C5 else SC5_PROXY[a][f])
        return E5[a][f]["shape_rms"], sd / np.sqrt(5)

    def ic20(a, f):
        m = E20[a][f]
        return abs(m["integral"]), np.hypot(m["boot_ic_std"],
                                            m["per_seed_ic_std"] / np.sqrt(m["n_members"]))

    def sc20(a, f):
        m = E20[a][f]
        proxy = np.std([x["shape_rms"] for x in C20[a][f]]) / np.sqrt(m["n_members"])
        return m["shape_rms"], np.hypot(m["boot_shape_rms_std"], proxy)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), sharex="col", sharey="col")
    rows = [("original sweep: 5-seed ensembles  [bars: stat $\\oplus$ $\\sigma_{seed}/\\sqrt{5}$]",
             FR5, ic5, sc5),
            ("extension: 20-seed ensembles, fractions $\\leq$ 0.1  [bars: paired-boot "
             "$\\oplus$ $\\sigma_{seed}/\\sqrt{K}$; $K{=}19$ where one member excluded]",
             FR20, ic20, sc20)]
    for r, (title, fracs, icf, scf) in enumerate(rows):
        for c, (fn, ylab, gate) in enumerate(
                [(icf, r"$|\mathrm{IC}|$  (normalisation error)", 0.01),
                 (scf, r"$\mathrm{SC_{rms}}$  (shape error)", 0.05)]):
            ax = axes[r][c]
            for k, a in enumerate(COL):
                v, e = zip(*[fn(a, f) for f in fracs])
                ax.errorbar(_x(fracs, k), v, yerr=e, marker="o", ms=5, lw=1.8, capsize=3,
                            color=COL[a], label=a if (r, c) == (0, 0) else None)
            ax.axhline(gate, ls=":", lw=1, color="grey", alpha=0.7)
            ax.text(0.0032, gate * 1.13, f"gate {gate}", fontsize=7.5, color="grey")
            _style(ax, ylab, xlab=(r == 1))
        axes[r][0].set_title(title, fontsize=9, loc="left")
    axes[0][0].legend(fontsize=9, frameon=False, loc="lower left")
    fig.suptitle("Ensemble ratio-closure vs training statistics, the two campaigns kept separate",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)


def fig_noise(E5, E20, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (E, fr, title, tkey) in zip(axes, [
            (E5, FR5, "K=5 (original sweep)", "integral_err"),
            (E20, FR20, "K=20 (extension)", "boot_ic_std")]):
        for k, a in enumerate(COL):
            seed = [E[a][f]["per_seed_ic_std"] / np.sqrt(E[a][f]["n_members"]) for f in fr]
            test = [E[a][f][tkey] for f in fr]
            ax.plot(_x(fr, k), seed, marker="o", ms=6, lw=1.8, color=COL[a],
                    label=f"{a}: seed term $\\sigma_{{seed}}/\\sqrt{{K}}$")
            ax.plot(_x(fr, k), test, marker="s", ms=6, mfc="white", mew=1.6, lw=1.2,
                    ls="--", color=COL[a], label=f"{a}: test floor")
        ax.axhline(0.01, ls=":", lw=1, color="grey", alpha=0.7)
        ax.text(0.0032, 0.0113, "gate 0.01", fontsize=7.5, color="grey")
        _style(ax, r"error on the ensemble $|\mathrm{IC}|$" if ax is axes[0] else None)
        ax.set_title(title, fontsize=10)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Where the calibration error bar comes from: training stochasticity (filled) "
                 "vs test-sample statistics (open)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    fig.savefig(out)
    plt.close(fig)


def fig_verdict(E20, out):
    import matplotlib.pyplot as plt
    COL2 = {"finetune": COL["finetune"], "frozen": COL["frozen"]}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)
    summ = {}
    for k, (a, dx) in enumerate({"finetune": 0.92, "frozen": 1.08}.items()):
        d, err = [], []
        for f in FR20:
            ma, ms = E20[a][f], E20["scratch"][f]
            d.append(abs(ms["integral"]) - abs(ma["integral"]))
            pair = np.std(np.abs(ma["boot_ic"]) - np.abs(ms["boot_ic"]))
            err.append(np.sqrt(pair ** 2 + ma["per_seed_ic_std"] ** 2 / ma["n_members"]
                               + ms["per_seed_ic_std"] ** 2 / ms["n_members"]))
        D, S = float(np.mean(d)), float(np.sqrt(np.sum(np.square(err))) / 5)
        summ[a] = (D, S)
        ax1.errorbar(np.array([float(f) for f in FR20]) * dx, d, yerr=err, marker="o", ms=5,
                     lw=1.8, capsize=3, color=COL2[a], label=a)
        ax1.errorbar([0.24 * 1.15 ** k], [D], yerr=[S], marker="s", ms=7, lw=3, capsize=4,
                     color=COL2[a])
    ax1.axhline(0, color="0.25", lw=1)
    ax1.axvline(0.17, color="0.85", lw=0.8)
    ax1.text(0.255, 0.035, "avg", ha="center", fontsize=8, color="0.35")
    for y, a, word in ((0.185, "frozen", "harms"), (0.155, "finetune", "bounded")):
        D, S = summ[a]
        ax1.text(0.0042, y, f"{a}: $\\Delta$={D:+.3f}$\\pm${S:.3f}  $\\Rightarrow$ {word} "
                 f"($z$={D / S:+.1f})", fontsize=8.5, color=COL2[a])
    ax1.set_ylim(-0.27, 0.21)
    _style(ax1, r"$|\mathrm{IC}|_{\rm scratch}-|\mathrm{IC}|_{\rm arm}$", log_y=False)
    ax1.set_title("calibration: pre-training does not help", fontsize=10)
    ax1.legend(loc="lower right", fontsize=8.5, frameon=False)
    for a, dx in {"finetune": 0.92, "frozen": 1.08}.items():
        d = [E20[a][f]["auc"] - E20["scratch"][f]["auc"] for f in FR20]
        e = [np.hypot(E20[a][f]["boot_auc_std"], E20["scratch"][f]["boot_auc_std"]) for f in FR20]
        ax2.errorbar(np.array([float(f) for f in FR20]) * dx, d, yerr=e, marker="o", ms=5,
                     lw=1.8, capsize=3, color=COL2[a], label=a)
    ax2.axhline(0, color="0.25", lw=1)
    ax2.annotate("optimiser-transient tier\n(known, K=5 study)", xy=(0.003, -0.009),
                 xytext=(0.0055, -0.0125), fontsize=7.5, color="0.35",
                 arrowprops=dict(arrowstyle="-", color="0.6", lw=0.8))
    _style(ax2, r"$\mathrm{AUC}_{\rm arm}-\mathrm{AUC}_{\rm scratch}$", log_y=False)
    ax2.set_title(r"discrimination: pre-training wins at every fraction $\geq$ 0.01", fontsize=10)
    ax2.legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.suptitle("Same 20-seed ensembles, same test events: opposite verdicts on the two axes "
                 "(up = pre-training better; bars: paired-bootstrap $\\oplus$ seed$/\\sqrt{K}$)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ens5", default=os.path.join(here, "results", "ensemble-k5.json"))
    ap.add_argument("--ens20", default=os.path.join(here, "results", "ensemble-k20.json"))
    ap.add_argument("--closure20", default=os.path.join(here, "results", "closure-k20.json"))
    ap.add_argument("--closure5", default=None)
    ap.add_argument("--outdir", default=os.path.join(here, "figures"))
    ap.add_argument("--png", action="store_true", help="also write PNG twins (not committed)")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    E5, E20, C20 = (json.load(open(p)) for p in (args.ens5, args.ens20, args.closure20))
    C5 = json.load(open(args.closure5)) if args.closure5 else None
    if C5 is None:
        print("note: K=5 SC seed term uses the embedded single-model spreads (pass --closure5)")
    os.makedirs(args.outdir, exist_ok=True)
    exts = ["pdf"] + (["png"] if args.png else [])
    for name, fn, a in (("auc_efficiency", fig_auc, (E5, E20)),
                        ("closure_trends", fig_closure, (E5, E20, C20, C5)),
                        ("noise_decomposition", fig_noise, (E5, E20)),
                        ("p5_verdict", fig_verdict, (E20,))):
        for ext in exts:
            fn(*a, os.path.join(args.outdir, f"{name}.{ext}"))
        print(f"wrote {args.outdir}/{name}.{'/'.join(exts)}")


if __name__ == "__main__":
    main()
