"""Closure TRENDS vs training fraction -- the data-efficiency view.

Gates answer "is this usable?"; the trend answers "how much data does each arm need to
reach a given quality?", which is the only question the FM has to win for NSBI. Plots
|IC| and shape closure against training fraction, one curve per arm, log-log, with the
gates drawn faintly for reference rather than as the subject.

Accepts either or both:
  --single <closure-trained.json>    {cfg: {size: [per-seed dicts]}}  -> mean +- s.e.m.
  --ensemble <ensemble-trained.json> {cfg: {size: metrics}}           -> one ensemble/cell

ERROR BARS (the point of this plot -- do not undersell them):
  ensemble |IC|   : stat (+) seed  =  sqrt(integral_err^2 + (per_seed_ic_std/sqrt(K))^2).
                    The stat floor alone is 3-10x too small (pull test 2026-07-17); the
                    training-stochasticity term dominates at every reduced fraction.
  ensemble SC_rms : no per-seed decomposition exists inside the ensemble file (SC is
                    nonlinear in the members), so the training-noise term is PROXIED by
                    the single-model seed spread / sqrt(K), taken from --single if given,
                    else from --sc-proxy {cfg: {size: sd}}, else bars are omitted and the
                    caption says so.

Usage:
  python plot_closure_trends.py --ensemble ensemble-trained.json \
         [--single closure-trained.json | --sc-proxy sc_sd.json] --output closure_trends.png
"""
import argparse
import json

import numpy as np

CONFIGS = ["finetune", "frozen", "scratch"]
COLORS = {"finetune": "#2F6BDE", "frozen": "#009988", "scratch": "#CC6611"}  # CVD-validated


def series(data, cfg, key, ensemble, sc_proxy=None):
    """-> (fractions, values, FULL errors)"""
    d = data.get(cfg, {})
    xs = sorted(d, key=float)
    if not xs:
        return [], [], []
    if ensemble:
        v, e = [], []
        for s in xs:
            m = d[s]
            k = max(m.get("n_members", 5), 1)
            if key == "integral":
                v.append(abs(m[key]))
                seed = m.get("per_seed_ic_std", 0.0) / np.sqrt(k)
                e.append(np.hypot(m["integral_err"], seed))
            else:
                v.append(m[key])
                proxy = (sc_proxy or {}).get(cfg, {}).get(s, np.nan) / np.sqrt(k)
                e.append(proxy if np.isfinite(proxy) else 0.0)
    else:
        v = [np.nanmean(np.abs([m[key] for m in d[s]])) for s in xs]
        e = [np.nanstd([abs(m[key]) for m in d[s]]) / np.sqrt(max(len(d[s]), 1)) for s in xs]
    return [float(s) for s in xs], np.asarray(v), np.asarray(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--single")
    ap.add_argument("--ensemble")
    ap.add_argument("--sc-proxy", help="{cfg: {size: seed-sd of single-model SC_rms}} "
                                       "for the ensemble SC error term")
    ap.add_argument("--output", default="closure_trends.png")
    args = ap.parse_args()
    if not (args.single or args.ensemble):
        ap.error("give --single and/or --ensemble")

    sc_proxy = None
    if args.single:                      # derive the SC training-noise proxy directly
        sing = json.load(open(args.single))
        sc_proxy = {c: {s: float(np.nanstd([m["shape_rms"] for m in sing[c][s]]))
                        for s in sing[c]} for c in sing}
    elif args.sc_proxy:
        sc_proxy = json.load(open(args.sc_proxy))

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
                xs, v, e = series(data, cfg, key, is_ens, sc_proxy)
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
    src = "stat $\\oplus$ $\\sigma_{seed}/\\sqrt{K}$" + \
          ("" if sc_proxy else "; SC bars omitted (no seed proxy given)")
    fig.suptitle("Ratio-closure trends vs training statistics — "
                 f"does pre-training shift the curves left?  [bars: {src}]", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
