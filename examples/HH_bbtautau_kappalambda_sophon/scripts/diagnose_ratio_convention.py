"""Which prior convention makes the EveNet score a density ratio? (G0.5 prerequisite)

The learned score p = sigma(logit) estimates a posterior under whatever class prior the
TRAINING loss effectively used. Converting it to a density ratio needs the matching
inverse prior:

    raw-weight training      r = p/(1-p) * W0/W1     (W_c = summed |event_weight| of class c)
    class-balanced training  r = p/(1-p)             (prior already equalised)
    unweighted training      r = p/(1-p) * N0/N1     (N_c = event counts)

Only the correct one satisfies the closure identity E_ref[r] = 1 (see eval_closure.py).
This script evaluates all three on a prediction.pt and reports E_ref[r] - 1 for each, so
the convention is chosen by measurement rather than assumption. Run it on the LARGEST
training fraction available: there the classifier is closest to converged, so a residual
offset is convention, not statistics.

Usage:
  python diagnose_ratio_convention.py --prediction <store>/predictions/<tag>/prediction.pt
  python diagnose_ratio_convention.py --store <store>            # auto-picks size1.0 runs
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_closure import _load_prediction, closure_metrics  # noqa: E402

_EPS = 1e-6


def conventions(p, y, w):
    p = np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)
    y = np.asarray(y) > 0.5
    w = np.abs(np.asarray(w, float))          # |w|: NLO signs cancel in the prior, not the sum
    odds = p / (1.0 - p)
    W1, W0 = w[y].sum(), w[~y].sum()
    N1, N0 = int(y.sum()), int((~y).sum())
    return {
        "raw-weight  (x W0/W1)": W0 / W1,
        "class-balanced (x 1) ": 1.0,
        "unweighted  (x N0/N1)": N0 / N1,
    }, odds, y, w, (W0, W1, N0, N1)


def report(path):
    score, label, weight = _load_prediction(path)
    factors, odds, y, w, (W0, W1, N0, N1) = conventions(score, label, weight)
    print(f"\n{os.path.basename(os.path.dirname(path))}")
    print(f"  test set: N0={N0} N1={N1} (N0/N1={N0/N1:.4f}) | "
          f"W0={W0:.1f} W1={W1:.1f} (W0/W1={W0/W1:.4f})")
    best, best_dev = None, np.inf
    for name, f in factors.items():
        r = odds * f
        ic = float(np.sum(w[~y] * r[~y]) / w[~y].sum() - 1.0)
        m = closure_metrics(score, y, w) if f == factors["raw-weight  (x W0/W1)"] else None
        extra = f"  SC_rms={m['shape_rms']:.4f}" if m else ""
        print(f"  {name}: factor={f:.4f}  E_ref[r]-1 = {ic:+.4f}{extra}")
        if abs(ic) < best_dev:
            best, best_dev = name, abs(ic)
    print(f"  -> closest to closure: {best} (|E_ref[r]-1| = {best_dev:.4f})")
    return best, best_dev


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prediction", help="a single prediction.pt")
    ap.add_argument("--store", help="scan <store>/predictions/*size1.0*/prediction.pt")
    args = ap.parse_args()

    paths = ([args.prediction] if args.prediction else
             sorted(glob.glob(os.path.join(args.store, "predictions", "*size1.0*",
                                           "prediction.pt"))))
    if not paths:
        raise SystemExit("no prediction.pt found")
    votes = {}
    for p in paths:
        best, dev = report(p)
        votes[best] = votes.get(best, 0) + 1
    print("\n=== verdict across", len(paths), "runs ===")
    for k, v in sorted(votes.items(), key=lambda t: -t[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
