"""Ensemble closure (G0.5 companion): are the per-model calibration offsets cancellable?

The sweep's per-seed integral-closure offsets scatter by +-0.07..0.27 while each run's own
statistical error is only ~0.02 -- i.e. the offsets are MODEL-specific and roughly zero
centred, not a common bias. Production NSBI never uses a single network for this reason
(the analysis config trains num_ensemble_members=5), and the kinematic ceiling it is
compared against is itself an ensemble (200 boosted trees). This script therefore forms
the like-for-like object: the per-(arm, fraction) ENSEMBLE ratio

    r_ens(x) = mean_k r_k(x)          (k = seeds; the ratio, not the score, is averaged
                                       because the ratio is what enters the likelihood)

and re-runs the identical closure metrics on it. If ensembling collapses |IC| toward its
statistical floor and lowers SC_norm, the single-model closure numbers are a property of
the training protocol rather than of the pre-trained representation, and the arm
comparison must be made at ensemble level.

Alignment: seeds must share the event ordering of the fixed test split, which is verified
per (arm, fraction) by comparing the label and weight vectors before averaging; misaligned
cells are skipped loudly rather than silently averaged.

Usage:
  python eval_ensemble.py --store_dir <store> [--output-prefix ensemble]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_closure import (_load_prediction, _prior_factor, closure_metrics,  # noqa: E402
                          load_norm)
from plot_data_efficiency import CONFIGS, _TAG, weighted_auc               # noqa: E402

_EPS = 1e-6


def cells(store):
    """-> {(cfg, size): [paths]}"""
    out = {}
    for p in sorted(glob.glob(os.path.join(store, "predictions", "*", "prediction.pt"))):
        m = _TAG.search(os.path.basename(os.path.dirname(p)))
        if m and m["cfg"] in CONFIGS:
            out.setdefault((m["cfg"], float(m["size"])), []).append(p)
    return out


def ensemble_cell(paths, use_abs_w=True, prior="balanced", norm=None):
    """-> (r_ens, y, w, per_seed_ic) or None if the seeds are not aligned."""
    ref_y = ref_w = None
    ratios, ics = [], []
    for p in paths:
        score, y, w = _load_prediction(p)
        y = np.asarray(y) > 0.5
        w = np.abs(np.asarray(w, float)) if use_abs_w else np.asarray(w, float)
        if ref_y is None:
            ref_y, ref_w = y, w
        elif len(y) != len(ref_y) or not np.array_equal(y, ref_y) \
                or not np.allclose(w, ref_w, rtol=1e-5):
            return None
        score = np.clip(score, _EPS, 1.0 - _EPS)
        r = score / (1.0 - score) * _prior_factor(prior, y, w, norm)
        ratios.append(r)
        ics.append(float(np.sum(w[~y] * r[~y]) / w[~y].sum() - 1.0))
    return np.mean(ratios, axis=0), ref_y, ref_w, np.asarray(ics)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store_dir", required=True)
    ap.add_argument("--output-prefix", default="ensemble")
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--neff-min", type=float, default=25.0)
    ap.add_argument("--normalization", help="evenet-train/normalization.pt -> use the "
                                            "'trained' prior + signed weights (matches "
                                            "eval_closure.py --normalization)")
    args = ap.parse_args()
    norm = load_norm(args.normalization) if args.normalization else None
    prior = "trained" if norm is not None else "balanced"
    use_abs = norm is None
    if norm is not None:
        print(f"prior factor = {_prior_factor('trained', None, None, norm):.4f}  "
              f"(convention: trained, signed weights)")

    results = {}
    for (cfg, size), paths in sorted(cells(args.store_dir).items()):
        got = ensemble_cell(paths, use_abs_w=use_abs, prior=prior, norm=norm)
        if got is None:
            print(f"SKIP {cfg} size={size}: seeds not aligned on the test split")
            continue
        r_ens, y, w, ics = got
        # closure_metrics works from a score; invert r -> p (monotone, prior already applied)
        p_ens = r_ens / (1.0 + r_ens)
        met = closure_metrics(p_ens, y, w, args.nbins, args.neff_min,
                              prior="balanced", use_abs_w=False)   # prior already in r_ens
        auc = weighted_auc(p_ens, y.astype(float), w)
        met.update(auc=auc, n_members=len(paths),
                   per_seed_ic_mean=float(np.mean(ics)),
                   per_seed_ic_absmean=float(np.mean(np.abs(ics))),
                   per_seed_ic_std=float(np.std(ics)))
        results.setdefault(cfg, {})[str(size)] = met
        print(f"{cfg:9s} size={size:<5} n={len(paths)}  AUC={auc:.4f}  "
              f"|IC|_ens={abs(met['integral']):.4f} (stat {met['integral_err']:.4f}) "
              f"vs per-seed |IC|={met['per_seed_ic_absmean']:.4f}"
              f"+-{met['per_seed_ic_std']:.4f} (signed mean {met['per_seed_ic_mean']:+.4f})  "
              f"SC_norm={met['shape_rms_norm']:.4f}  chi2/ndf={met['shape_chi2ndf']:.2f}")

    with open(f"{args.output_prefix}.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.output_prefix}.json")


if __name__ == "__main__":
    main()
