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

--bootstrap N adds an H1-style Poisson(mu=1) test-set bootstrap per cell: the ensemble
score is FIXED (no retraining) and only the test sample fluctuates, so boot_*_std is the
pure test-statistical component -- the complement of per_seed_ic_std, which is training
stochasticity at fixed test set. The two add in quadrature to the full error bar.

Usage:
  python eval_ensemble.py --store_dir <store> [--output-prefix ensemble] [--bootstrap 200]
"""
import argparse
import glob
import json
import os
import sys
import zlib

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
        try:
            score, y, w = _load_prediction(p)
        except Exception as e:                    # noqa: BLE001
            print(f"  skip member {p}: {e}")
            continue
        if not np.all(np.isfinite(np.asarray(score, float))):
            n_bad = int((~np.isfinite(np.asarray(score, float))).sum())
            print(f"  EXCLUDING member {p}: {n_bad} non-finite scores "
                  f"(diverged run? check its training log) -- K reduced for this cell")
            continue
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


def bootstrap_cell(p_ens, y, w, nbins, neff_min, n_boot, tag, keep_bins):
    """Poisson(mu=1) resampling of the test set at fixed ensemble score -> test-stat spread.

    keep_bins (the point estimate's bins_kept) freezes the shape statistic's bin set:
    Poisson(1) halves each bin's Kish n_eff, so re-deriving the acceptance per replica
    would drop marginal bins and measure a different statistic than the one quoted.

    tag seeds the rng and is the SIZE only, not cfg-size: all arms share the test
    events, so common replica draws make the per-replica ICs (stored as boot_ic)
    PAIRED across arms -- std over replicas of IC_A - IC_B cancels the common
    test-sample fluctuation. Pairing is valid iff the arms share event order; compare
    boot_w_checksum across cells before differencing.
    """
    rng = np.random.default_rng(zlib.crc32(tag.encode()))
    ics, shapes, aucs = [], [], []
    for _ in range(n_boot):
        wb = w * rng.poisson(1.0, size=len(w))
        m = closure_metrics(p_ens, y, wb, nbins, neff_min, prior="balanced",
                            use_abs_w=False, keep_bins=keep_bins)
        ics.append(m["integral"])
        shapes.append(m["shape_rms"])
        aucs.append(weighted_auc(p_ens, y.astype(float), wb))
    return {"n_boot": n_boot,
            "boot_ic_mean": float(np.mean(ics)),
            "boot_ic_std": float(np.std(ics)),
            "boot_shape_rms_std": float(np.std(shapes)),
            "boot_auc_std": float(np.std(aucs)),
            "boot_ic": [round(float(v), 6) for v in ics],
            "boot_w_checksum": round(float(np.sum(w)), 4)}


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
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="Poisson(1) test-set bootstrap replicas (e.g. 200)")
    args = ap.parse_args()
    norm = load_norm(args.normalization) if args.normalization else None
    prior = "trained" if norm is not None else "balanced"
    use_abs = norm is None
    if norm is not None:
        print(f"prior factor = {_prior_factor('trained', None, None, norm):.4f}  "
              f"(convention: trained, signed weights)")

    results = {}
    for (cfg, size), paths in sorted(cells(args.store_dir).items()):
        print(f"cell {cfg} size={size}: loading {len(paths)} members ...", flush=True)
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
        met.update(auc=auc, n_members=len(ics),
                   per_seed_ic_mean=float(np.mean(ics)),
                   per_seed_ic_absmean=float(np.mean(np.abs(ics))),
                   per_seed_ic_std=float(np.std(ics)))
        results.setdefault(cfg, {})[str(size)] = met
        print(f"{cfg:9s} size={size:<5} n={len(ics)}  AUC={auc:.4f}  "
              f"|IC|_ens={abs(met['integral']):.4f} (stat {met['integral_err']:.4f}) "
              f"vs per-seed |IC|={met['per_seed_ic_absmean']:.4f}"
              f"+-{met['per_seed_ic_std']:.4f} (signed mean {met['per_seed_ic_mean']:+.4f})  "
              f"SC_norm={met['shape_rms_norm']:.4f}  chi2/ndf={met['shape_chi2ndf']:.2f}")
        if args.bootstrap:
            met.update(bootstrap_cell(p_ens, y, w, args.nbins, args.neff_min,
                                      args.bootstrap, f"size-{size}",
                                      met["bins_kept"]))
            seed_term = met["per_seed_ic_std"] / np.sqrt(len(ics))
            print(f"{'':9s} boot({args.bootstrap}): IC +-{met['boot_ic_std']:.4f} (test) "
                  f"(+) +-{seed_term:.4f} (seed/sqrtK) = "
                  f"+-{float(np.hypot(met['boot_ic_std'], seed_term)):.4f} total   "
                  f"SC_rms +-{met['boot_shape_rms_std']:.4f}   AUC +-{met['boot_auc_std']:.4f}")

    with open(f"{args.output_prefix}.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.output_prefix}.json")


if __name__ == "__main__":
    main()
