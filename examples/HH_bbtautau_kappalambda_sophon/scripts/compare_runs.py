"""Compare two sets of predictions run-for-run: are they the same model, or not?

Used to TEST a claim rather than assume it. The source audit concluded that flipping
`noise_prob` cannot change these runs (it is read only inside a `generation` branch that
three config flags disable, while the branch that produced our scores sets full_time=0).
That is falsifiable: if the ablation predictions differ beyond seed-level noise, the audit
is wrong. This script quantifies the difference on the shared test split.

For each tag present in BOTH stores it reports:
  max|dp|      largest per-event score difference   (0 => bit-identical outputs)
  rms dp       RMS score difference
  dAUC, dIC    metric-level differences
and classifies each pair as IDENTICAL / NEGLIGIBLE / DIFFERENT.

Usage:
  python compare_runs.py --a <storeA> --b <storeB> [--tag-filter size1.0]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_closure import _load_prediction, closure_metrics   # noqa: E402
from plot_data_efficiency import weighted_auc                # noqa: E402


def tags(store, filt):
    out = {}
    for p in sorted(glob.glob(os.path.join(store, "predictions", "*", "prediction.pt"))):
        t = os.path.basename(os.path.dirname(p))
        if not filt or filt in t:
            out[t] = p
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="store dir A (e.g. the original sweep)")
    ap.add_argument("--b", required=True, help="store dir B (e.g. the ablation)")
    ap.add_argument("--tag-filter", default="", help="substring filter, e.g. size1.0")
    args = ap.parse_args()

    A, B = tags(args.a, args.tag_filter), tags(args.b, args.tag_filter)
    common = sorted(set(A) & set(B))
    print(f"A: {len(A)} runs   B: {len(B)} runs   common: {len(common)}")
    if not common:
        print("no common tags -- check paths / tag naming")
        for t in list(A)[:3]:
            print("  A example:", t)
        for t in list(B)[:3]:
            print("  B example:", t)
        return
    verdicts = {}
    for t in common:
        pa, ya, wa = _load_prediction(A[t])
        pb, yb, wb = _load_prediction(B[t])
        if len(pa) != len(pb) or not np.array_equal(np.asarray(ya), np.asarray(yb)):
            print(f"{t}: SKIP (test sets differ: {len(pa)} vs {len(pb)} events)")
            continue
        d = pa - pb
        mx, rms = float(np.abs(d).max()), float(np.sqrt((d ** 2).mean()))
        dauc = weighted_auc(pa, ya, wa) - weighted_auc(pb, yb, wb)
        dic = closure_metrics(pa, ya, wa)["integral"] - closure_metrics(pb, yb, wb)["integral"]
        v = "IDENTICAL" if mx == 0 else ("NEGLIGIBLE" if mx < 1e-4 else "DIFFERENT")
        verdicts[v] = verdicts.get(v, 0) + 1
        print(f"{t:48s} max|dp|={mx:.3e} rms={rms:.3e} dAUC={dauc:+.5f} dIC={dic:+.5f}  {v}")
    print("\n=== verdict ===")
    for k, n in sorted(verdicts.items()):
        print(f"  {k}: {n}")
    print("  IDENTICAL/NEGLIGIBLE across the board => the flipped knob is inert (audit confirmed).")
    print("  DIFFERENT => the audit is wrong; the knob does reach the classification path.")


if __name__ == "__main__":
    main()
