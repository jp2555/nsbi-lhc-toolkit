"""Kinematic CEILING from the prelim-result FEATURE ntuple (dihiggs_powheg_data.root).

The CMS full-sim bbtautau files at /pscratch/sd/j/jing/NSBI-irishep/dihiggs_bbtautau/ are
HIGH-LEVEL feature ntuples (trees tree_sbi_lam{0,1,2p45,5}; 12 features + weights), NOT
object-level events -- they CANNOT feed EveNet (no per-object four-vectors to build the
token cloud from). What they are exactly right for is the G0 decision rule's kinematic
ceiling: the same features behind the published unbinned NSBI result.

This trains a HistGradientBoosting classifier ref-tree vs hyp-tree at the sweep's dataset
fractions x seeds, on a FIXED held-out test split, and writes:
  <prefix>.json     {size: mean weighted AUC}  -- plot_data_efficiency.py --ceiling format
  <prefix>_details.json                        -- per (size, seed) AUC + closure metrics
Closure metrics (eval_closure.closure_metrics) come along for free, so the ceiling also
anchors the G0.5 comparison: a feature-level classifier's closure at each budget.

Conventions: training AND evaluation weights = |w| class-normalized. Weighted AUC is
invariant to per-class scales, and the balanced convention makes closure measure
calibration proper (the raw signed sums differ by the kl cross sections -- W0/W1 ~ 0.35
for lam1/lam5 -- which would swamp IC with a physics normalization, not a bias).
+-inf feature values (deta_hh carries inf sentinels in ~10% of events) are mapped to
NaN, which HistGradientBoosting routes natively. CPU-only (numpy + sklearn + uproot),
fine on a login node.

Usage (Perlmutter):
  python3 scripts/feature_ceiling.py \
      --input /pscratch/sd/j/jing/NSBI-irishep/dihiggs_bbtautau/dihiggs_powheg_data.root \
      --tree-ref tree_sbi_lam1 --tree-hyp tree_sbi_lam5 --out-prefix ceiling-kl5
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_closure import closure_metrics          # noqa: E402
from plot_data_efficiency import weighted_auc     # noqa: E402

FEATURES = ["m_hh", "m_bb", "m_tautau", "pt_hh", "dr_bb", "dr_tautau",
            "pt_h1", "pt_h2", "log_m_hh", "deta_hh", "dphi_hh", "cos_theta_star"]
_SPLIT_SEED = 123                                 # fixed test split, shared by all points


def load_tree(path, tree, features, weights_branch, max_events=0):
    import uproot
    t = uproot.open(f"{path}:{tree}")
    arr = t.arrays(features + [weights_branch], library="np")
    X = np.stack([arr[f] for f in features], axis=1).astype(np.float64)
    X[np.isinf(X)] = np.nan            # inf sentinels -> NaN (HistGB handles natively)
    w = arr[weights_branch].astype(np.float64)
    if max_events:
        X, w = X[:max_events], w[:max_events]
    return X, w


def split_train_test(X, w, test_frac):
    idx = np.random.default_rng(_SPLIT_SEED).permutation(len(w))
    n_te = int(len(w) * test_frac)
    te, tr = idx[:n_te], idx[n_te:]
    return (X[tr], w[tr]), (X[te], w[te])


def run(args):
    X0, w0 = load_tree(args.input, args.tree_ref, args.features, args.weights_branch,
                       args.max_events)
    X1, w1 = load_tree(args.input, args.tree_hyp, args.features, args.weights_branch,
                       args.max_events)
    print(f"{args.tree_ref}: {len(w0)} events ({(w0 < 0).mean():.1%} negative weights)  "
          f"{args.tree_hyp}: {len(w1)} events ({(w1 < 0).mean():.1%})")
    (X0tr, w0tr), (X0te, w0te) = split_train_test(X0, w0, args.test_frac)
    (X1tr, w1tr), (X1te, w1te) = split_train_test(X1, w1, args.test_frac)

    X_te = np.concatenate([X0te, X1te])
    y_te = np.concatenate([np.zeros(len(w0te)), np.ones(len(w1te))])
    wa_te, wb_te = np.abs(w0te), np.abs(w1te)     # |w|, class-normalized (as in training)
    w_te = np.concatenate([wa_te / wa_te.sum(), wb_te / wb_te.sum()])

    from sklearn.ensemble import HistGradientBoostingClassifier
    details, ceiling = {}, {}
    for size in args.sizes:
        aucs = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            i0 = rng.choice(len(w0tr), max(2, int(len(w0tr) * size)), replace=False)
            i1 = rng.choice(len(w1tr), max(2, int(len(w1tr) * size)), replace=False)
            Xtr = np.concatenate([X0tr[i0], X1tr[i1]])
            ytr = np.concatenate([np.zeros(len(i0)), np.ones(len(i1))])
            wa, wb = np.abs(w0tr[i0]), np.abs(w1tr[i1])   # |w|, class-normalized
            wtr = np.concatenate([wa / wa.sum(), wb / wb.sum()])

            clf = HistGradientBoostingClassifier(
                max_iter=args.max_iter, early_stopping=False, random_state=seed)
            clf.fit(Xtr, ytr, sample_weight=wtr)
            p = clf.predict_proba(X_te)[:, 1]

            auc = weighted_auc(p, y_te, w_te)
            met = closure_metrics(p, y_te, w_te)
            aucs.append(auc)
            details.setdefault(str(size), []).append(dict(seed=seed, auc=auc, **met))
        ceiling[str(size)] = float(np.nanmean(aucs))
        d = details[str(size)]
        print(f"size={size:<5} AUC={np.nanmean(aucs):.4f}+-{np.nanstd(aucs):.4f}  "
              f"|IC|={np.nanmean([abs(m['integral']) for m in d]):.4f}  "
              f"SC_rms={np.nanmean([m['shape_rms'] for m in d]):.4f}  (n={len(d)})")

    with open(f"{args.out_prefix}.json", "w") as f:
        json.dump(ceiling, f, indent=1)
    with open(f"{args.out_prefix}_details.json", "w") as f:
        json.dump(details, f, indent=1)
    print(f"wrote {args.out_prefix}.json (+_details.json)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="feature ntuple ROOT file")
    ap.add_argument("--tree-ref", default="tree_sbi_lam1", help="class-0 tree (SM reference)")
    ap.add_argument("--tree-hyp", default="tree_sbi_lam5", help="class-1 tree (hypothesis)")
    ap.add_argument("--features", nargs="+", default=FEATURES)
    ap.add_argument("--weights-branch", default="weights")
    ap.add_argument("--sizes", nargs="+", type=float, default=[0.01, 0.03, 0.1, 0.3, 1.0])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--max-events", type=int, default=0, help="cap per tree (0 = all)")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--out-prefix", default="ceiling")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
