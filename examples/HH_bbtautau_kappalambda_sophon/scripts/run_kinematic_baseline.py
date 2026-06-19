"""Kinematic CEILING baseline for the kappa_lambda diagnostic.

How separable are two kappa_lambda points using high-level *event kinematics* alone?
This trains a gradient-boosted tree on ~14 physics features (m_HH proxies, HT, jet
masses, MET, ...) reconstructed from the SAME cloud .npz the constituent models use,
and reports validation weighted-BCE + AUC at each statistics point.

Purpose: a ceiling to judge the constituent models. kl0-vs-kl5 is a pure-kinematics
task (same final state, different m_HH shape). If this simple BDT on hand-built
kinematics clearly BEATS the scratch/sophon constituent models, then those models are
leaving the kinematic signal on the table (and the fix is feeding kinematics, not more
pretraining). If it's comparable, ~0.64 BCE is just how separable kl0/kl5 are.

Runs on CPU (numpy + sklearn only; no torch). Reuses the exact same events, class-
normalized weights, and shuffling as run_kappa_diagnostic so the numbers overlay
directly on ablation.json / ablation.png.

Usage:
  pixi run -e nsbi-env python run_kinematic_baseline.py \
      --clouds-dir $SCRATCH/dihiggs/clouds_subset \
      --point-a kl5 --point-b kl0 --sizes 2000 5000 \
      --out-dir baseline_kl0_vs_kl5
"""
import argparse
import json
import os

import numpy as np

_CLOUD_KEYS = ("parts", "part_mask", "jet_mask", "obj", "obj_mask")

# --- data helpers: mirror run_kappa_diagnostic exactly, duplicated here so this
#     script stays torch-free (numpy + sklearn only) and runs on the CPU env. ---


def load_point(npz_path, num=0):
    d = np.load(npz_path)
    n = d["w"].shape[0]
    if num:
        n = min(num, n)
    return {k: d[k][:n] for k in (_CLOUD_KEYS + ("w",))}


def _shuffle(d, seed=0):
    n = d["w"].shape[0]
    idx = np.random.default_rng(seed).permutation(n)
    return {k: v[idx] for k, v in d.items()}


def build_binary_task(clouds_a, clouds_b):
    """class 0 = a, class 1 = b; weights = per-class-normalized |w| (same as the
    constituent diagnostic, so the val-BCE is directly comparable)."""
    out = {k: np.concatenate([clouds_a[k], clouds_b[k]], axis=0) for k in _CLOUD_KEYS}
    na, nb = clouds_a["w"].shape[0], clouds_b["w"].shape[0]
    wa, wb = np.abs(clouds_a["w"]).astype(np.float64), np.abs(clouds_b["w"]).astype(np.float64)
    wa = wa / wa.sum() if wa.sum() else wa
    wb = wb / wb.sum() if wb.sum() else wb
    out["y"] = np.concatenate([np.zeros(na), np.ones(nb)]).astype(np.int64)
    out["w"] = np.concatenate([wa, wb]).astype(np.float64)
    return out


# --- high-level kinematic feature extraction (all vectorized over events) ---


def _inv_mass(p4):
    """invariant mass of a (..., 4) [px, py, pz, E] array."""
    px, py, pz, E = p4[..., 0], p4[..., 1], p4[..., 2], p4[..., 3]
    return np.sqrt(np.clip(E * E - (px * px + py * py + pz * pz), 0.0, None))


def _eta_phi(v):
    pt = np.hypot(v[..., 0], v[..., 1])
    eta = np.arcsinh(np.divide(v[..., 2], pt, out=np.zeros_like(pt), where=pt > 0))
    phi = np.arctan2(v[..., 1], v[..., 0])
    return eta, phi


FEATURE_NAMES = [
    "n_jets", "HT", "lead_jet_pt", "sublead_jet_pt", "m_jj_lead2", "dR_jj_lead2",
    "m_alljets", "pT_alljets", "m_vis", "pT_vis", "n_leptons", "sum_lepton_pt",
    "MET", "m_eff",
]


def high_level_features(clouds):
    """Reconstruct ~14 event-level kinematics from constituent 4-vectors + object tokens.

    Jets carry the b / hadronic-tau legs (constituent (px,py,pz,E) in parts cols 17:21);
    objects carry isolated e/mu (type 1/2) and MET (type 3). m_vis / m_eff are the
    kappa_lambda-sensitive m_HH proxies (exact m_HH needs the tau neutrinos)."""
    parts, pm, jm = clouds["parts"], clouds["part_mask"], clouds["jet_mask"]
    obj, om = clouds["obj"], clouds["obj_mask"]

    # jet 4-vectors = sum of constituent 4-vectors (cols 17:21), per jet, masked
    jet4 = (parts[..., 17:21] * pm[..., None]).sum(axis=2) * jm[..., None]   # (N, J, 4)
    jpt = np.hypot(jet4[..., 0], jet4[..., 1])                               # (N, J)
    n_jets = jm.sum(1)
    HT = jpt.sum(1)
    lead_pt, sub_pt = jpt[:, 0], jpt[:, 1]            # jets are pt-sorted in the converter

    lead2 = jet4[:, 0] + jet4[:, 1]
    m_jj = _inv_mass(lead2)
    e0, p0 = _eta_phi(jet4[:, 0])
    e1, p1 = _eta_phi(jet4[:, 1])
    dphi = np.arctan2(np.sin(p0 - p1), np.cos(p0 - p1))
    dR_jj = np.hypot(e0 - e1, dphi)

    alljet = jet4.sum(1)
    m_alljets = _inv_mass(alljet)
    pT_alljets = np.hypot(alljet[..., 0], alljet[..., 1])

    # objects: leptons (type 1/2) -> 4-vectors; MET (type 3) -> magnitude only
    tid = obj[..., 5]
    lep = om * ((tid == 1) | (tid == 2))
    met = om * (tid == 3)
    lpt = np.exp(obj[..., 0])
    leta, lsin, lcos = obj[..., 1], obj[..., 2], obj[..., 3]
    lpx = (lpt * lcos * lep).sum(1)
    lpy = (lpt * lsin * lep).sum(1)
    lpz = (lpt * np.sinh(leta) * lep).sum(1)
    lE = (lpt * np.cosh(leta) * lep).sum(1)
    lep4 = np.stack([lpx, lpy, lpz, lE], axis=1)
    n_lep = lep.sum(1)
    sum_lep_pt = (lpt * lep).sum(1)
    metmag = (obj[..., 4] * met).sum(1)

    vis = alljet + lep4
    m_vis = _inv_mass(vis)
    pT_vis = np.hypot(vis[..., 0], vis[..., 1])
    m_eff = HT + sum_lep_pt + metmag

    X = np.stack([n_jets, HT, lead_pt, sub_pt, m_jj, dR_jj, m_alljets, pT_alljets,
                  m_vis, pT_vis, n_lep, sum_lep_pt, metmag, m_eff], axis=1)
    return np.nan_to_num(X).astype(np.float64)


def evaluate(X, y, w, seed=0):
    """Fit a gradient-boosted tree; return (val weighted-BCE, val weighted-AUC)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss, roc_auc_score
    Xtr, Xv, ytr, yv, wtr, wv = train_test_split(
        X, y, w, test_size=0.3, random_state=seed, stratify=y)
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=None,
        early_stopping=True, validation_fraction=0.15, random_state=seed)
    clf.fit(Xtr, ytr, sample_weight=wtr)
    p = np.clip(clf.predict_proba(Xv)[:, 1], 1e-9, 1.0 - 1e-9)
    ll = log_loss(yv, p, sample_weight=wv, labels=[0, 1])
    auc = roc_auc_score(yv, p, sample_weight=wv)
    return float(ll), float(auc)


def run(clouds_dir, point_a, point_b, sizes, out_dir, repeats=1):
    os.makedirs(out_dir, exist_ok=True)
    nmax = max(sizes)
    a_full = _shuffle(load_point(os.path.join(clouds_dir, f"{point_a}.npz"), nmax))
    b_full = _shuffle(load_point(os.path.join(clouds_dir, f"{point_b}.npz"), nmax))
    curve = {}
    for N in sorted(sizes):
        a = {k: v[:N] for k, v in a_full.items()}
        b = {k: v[:N] for k, v in b_full.items()}
        task = build_binary_task(a, b)
        X = high_level_features(task)
        lls, aucs = [], []
        for r in range(repeats):
            ll, auc = evaluate(X, task["y"], task["w"], seed=r)
            lls.append(ll)
            aucs.append(auc)
        curve[N] = {"bce": float(np.mean(lls)), "bce_std": float(np.std(lls)),
                    "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                    "n_events": int(task["y"].shape[0])}
        s = f" +/- {curve[N]['bce_std']:.4f}" if repeats > 1 else ""
        print(f"[baseline] N={N}/point: BCE={curve[N]['bce']:.4f}{s}  AUC={curve[N]['auc']:.4f}")
    with open(os.path.join(out_dir, "baseline.json"), "w") as fh:
        json.dump(curve, fh, indent=2)
    _plot(curve, sorted(sizes), out_dir, f"{point_b} vs {point_a}")
    return curve


def _plot(curve, sizes, out_dir, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                       # pragma: no cover
        print(f"[plot] skipped ({exc})")
        return
    xs = [N for N in sizes if N in curve]
    bce = [curve[N]["bce"] for N in xs]
    auc = [curve[N]["auc"] for N in xs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(xs, bce, marker="o", color="green", label="high-level BDT")
    ax1.axhline(0.6931, ls="--", color="gray", lw=1, label="random (ln 2)")
    ax1.set_xscale("log")
    ax1.set_xlabel("events per point (N)")
    ax1.set_ylabel("validation weighted-BCE")
    ax1.set_title(f"kinematic ceiling: {title}")
    ax1.legend()
    ax2.plot(xs, auc, marker="o", color="green")
    ax2.axhline(0.5, ls="--", color="gray", lw=1)
    ax2.set_xscale("log")
    ax2.set_xlabel("events per point (N)")
    ax2.set_ylabel("validation weighted-AUC")
    ax2.set_title("separation power")
    fig.tight_layout()
    path = os.path.join(out_dir, "baseline.png")
    fig.savefig(path, dpi=120)
    print(f"[plot] -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clouds-dir", required=True)
    ap.add_argument("--point-a", default="kl5", help="reference point label")
    ap.add_argument("--point-b", default="kl0", help="other point label")
    ap.add_argument("--sizes", nargs="*", type=int, default=[2000, 5000, 10000, 20000, 50000, 100000],
                    help="events per point to evaluate (matches the ablation sweep)")
    ap.add_argument("--repeats", type=int, default=1, help="seeds per point -> mean +/- std")
    ap.add_argument("--out-dir", default="baseline_out")
    args = ap.parse_args()
    run(args.clouds_dir, args.point_a, args.point_b, args.sizes, args.out_dir,
        repeats=args.repeats)


if __name__ == "__main__":
    main()
