"""Which prior convention makes the EveNet score a density ratio? (G0.5 prerequisite)

The learned score p = sigma(logit) estimates a posterior under whatever class prior the
TRAINING loss effectively used. Converting it to a density ratio needs the matching
inverse prior:

    raw-weight training      r = p/(1-p) * W0/W1     (W_c = summed |event_weight| of class c)
    class-balanced training  r = p/(1-p)             (prior already equalised)
    unweighted training      r = p/(1-p) * N0/N1     (N_c = event counts)
    EXACT (--normalization)  r = p/(1-p) * (c0/c1) * (W0/W1)

The EXACT row is the one the source actually implements. EveNet's classification loss is
    sum_i c_{y_i} w_i CE_i / sum_i c_{y_i} w_i
(evenet/network/loss/classification.py; c = normalization.pt["class_balance"], built by
postprocess.py from effective-number-of-samples reweighting of the WEIGHTED class counts,
so it is NOT inverse-frequency). Its population optimum is
    p/(1-p) = (c1/c0) * dW1(x)/dW0(x),
hence r(x) = p/(1-p) * (c0/c1) * (W0/W1). The first three rows all omit c entirely, which
is why they leave a same-sign residual in every arm, seed and fraction.

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


def load_class_balance(path):
    """-> (class_balance, class_counts) from normalization.pt, or (None, None)."""
    import torch
    d = torch.load(path, map_location="cpu")
    if not isinstance(d, dict):
        return None, None
    cb = d.get("class_balance")
    cc = d.get("class_counts", d.get("class_count"))
    to_np = lambda v: None if v is None else np.asarray(
        v.detach().cpu().numpy() if hasattr(v, "detach") else v, dtype=float)
    print(f"  normalization.pt keys: {sorted(d)}")
    return to_np(cb), to_np(cc)


def conventions(p, y, w, class_balance=None):
    p = np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)
    y = np.asarray(y) > 0.5
    w = np.abs(np.asarray(w, float))          # |w|: NLO signs cancel in the prior, not the sum
    odds = p / (1.0 - p)
    W1, W0 = w[y].sum(), w[~y].sum()
    N1, N0 = int(y.sum()), int((~y).sum())
    facs = {
        "raw-weight  (x W0/W1)": W0 / W1,
        "class-balanced (x 1) ": 1.0,
        "unweighted  (x N0/N1)": N0 / N1,
    }
    if class_balance is not None and len(class_balance) >= 2:
        c0, c1 = float(class_balance[0]), float(class_balance[1])
        facs[f"EXACT c0/c1*W0/W1   "] = (c0 / c1) * (W0 / W1)
        facs[f"class_balance only  "] = c0 / c1
    return facs, odds, y, w, (W0, W1, N0, N1)


def report(path, class_balance=None):
    score, label, weight = _load_prediction(path)
    factors, odds, y, w, (W0, W1, N0, N1) = conventions(score, label, weight, class_balance)
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
    # the factor this run WOULD need for exact closure: if a single constant reproduces
    # it across arms/seeds, the residual is a convention error, not model miscalibration
    need = 1.0 / float(np.sum(w[~y] * odds[~y]) / w[~y].sum())
    print(f"  -> closest: {best} (|E_ref[r]-1| = {best_dev:.4f}) | "
          f"factor needed for exact closure: {need:.4f}")
    return best, best_dev, need


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prediction", help="a single prediction.pt")
    ap.add_argument("--store", help="scan <store>/predictions/*size1.0*/prediction.pt")
    ap.add_argument("--normalization", help="path to evenet-train/normalization.pt -> adds the "
                                            "EXACT class_balance-based convention")
    args = ap.parse_args()

    paths = ([args.prediction] if args.prediction else
             sorted(glob.glob(os.path.join(args.store, "predictions", "*size1.0*",
                                           "prediction.pt"))))
    if not paths:
        raise SystemExit("no prediction.pt found")
    cb = None
    if args.normalization:
        cb, cc = load_class_balance(args.normalization)
        print(f"  class_balance = {cb}\n  class_counts  = {cc}")
        if cb is not None and len(cb) >= 2:
            print(f"  c0/c1 = {cb[0]/cb[1]:.4f}")
    votes, needs = {}, []
    for p in paths:
        best, dev, need = report(p, cb)
        votes[best] = votes.get(best, 0) + 1
        needs.append(need)
    print("\n=== verdict across", len(paths), "runs ===")
    for k, v in sorted(votes.items(), key=lambda t: -t[1]):
        print(f"  {k}: {v}")
    needs = np.asarray(needs)
    print(f"  factor needed for exact closure: mean {needs.mean():.4f} "
          f"+- {needs.std():.4f}  (range {needs.min():.4f}-{needs.max():.4f})")
    print("  -> a SMALL spread means one constant convention factor explains the residual;"
          "\n     a large spread means genuine per-model miscalibration remains.")


if __name__ == "__main__":
    main()
