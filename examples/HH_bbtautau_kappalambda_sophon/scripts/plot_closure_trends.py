"""Closure TRENDS vs training fraction -- the data-efficiency view.

Gates answer "is this usable?"; the trend answers "how much data does each arm need to
reach a given quality?", which is the only question the FM has to win for NSBI. Plots
|IC| and shape closure against training fraction, one curve per arm, log-log, with the
gates drawn faintly for reference rather than as the subject.

Accepts either or both:
  --single <closure-trained.json>    {cfg: {size: [per-seed dicts]}}  -> mean +- s.e.m.
  --ensemble <ensemble-trained.json> {cfg: {size: metrics}}           -> single point/cell

Usage:
  python plot_closure_trends.py --ensemble ensemble-trained.json [--single closure-trained.json] \
         --output closure_trends.png
"""
import argparse
import json

import numpy as np

CONFIGS = ["finetune", "frozen", "scratch"]
COLORS = {"finetune": "#1f77b4", "frozen": "#2ca02c", "scratch": "#d62728"}


def series(data, cfg, key, ensemble):
    """-> (fractions, values, errors)"""
    d = data.get(cfg, {})
    xs = sorted(d, key=float)
    if not xs:
        return [], [], []
    if ensemble:
        v = [abs(d[s][key]) if key == "integral" else d[s][key] for s in xs]
        e = [d[s]["integral_err"] if key == "integral" else 0.0 for s in xs]
    else:
        v = [np.nanmean(np.abs([m[key] for m in d[s]])) for s in xs]
        e = [np.nanstd([abs(m[key]) for m in d[s]]) / np.sqrt(max(len(d[s]), 1)) for s in xs]
    return [float(s) for s in xs], np.asarray(v), np.asarray(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--single")
    ap.add_argument("--ensemble")
    ap.add_argument("--output", default="closure_trends.png")
    args = ap.parse_args()
    if not (args.single or args.ensemble):
        ap.error("give --single and/or --ensemble")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("integral", r"$|\mathrm{IC}|$  (normalisation error)", 0.01),
              ("shape_rms", r"$\mathrm{SC_{rms}}$  (shape error)", 0.05)]
    sources = [(p, e) for p, e in ((args.single, False), (args.ensemble, True)) if p]
    fig, axes = plt.subplots(len(sources), 2, figsize=(11, 4.4 * len(sources)),
                             squeeze=False)

    for r, (path, is_ens) in enumerate(sources):
        data = json.load(open(path))
        label = "5-seed ensemble" if is_ens else "single model (mean $\\pm$ s.e.m.)"
        for c, (key, ylab, gate) in enumerate(panels):
            ax = axes[r][c]
            for cfg in CONFIGS:
                xs, v, e = series(data, cfg, key, is_ens)
                if not xs:
                    continue
                ax.errorbar(xs, v, yerr=e if np.any(e) else None, marker="o", ms=5,
                            capsize=3, lw=1.8, color=COLORS[cfg], label=cfg)
            ax.axhline(gate, ls=":", lw=1, color="grey", alpha=0.7)
            ax.text(0.0032, gate * 1.12, f"gate {gate}", fontsize=7, color="grey")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("training fraction")
            ax.set_ylabel(ylab)
            ax.set_title(f"{label}", fontsize=10)
            ax.grid(alpha=0.25, which="both", lw=0.4)
            if c == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Ratio-closure trends vs training statistics — "
                 "does pre-training shift the curves left?", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
