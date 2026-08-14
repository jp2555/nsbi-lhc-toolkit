"""Closure Plot (G0.5): density-ratio calibration vs training fraction, per pretrain config.

AUC orders events; NSBI needs the CALIBRATED ratio. Neither arXiv:2606.14870 (rejection
metrics only) nor arXiv:2601.17126 (SIC only) tests whether a fine-tuned FM classifier
converts to an unbiased density ratio -- this script measures exactly that on the same
prediction.pt files the AUC money plot uses (see EVENET_FEASIBILITY_NOTE sec 8, gate G0.5).

For scores p, labels y (0=ref, 1=hyp), weights w, the ratio estimate is
    r_hat(x) = p/(1-p) * f_prior
with f_prior fixed by the convention EveNet's own loss implies (see _prior_factor;
pass --normalization <evenet-train/normalization.pt> to use it). Two metrics per
(config, size, seed):

  integral closure   IC = sum_{y=0} w * r_hat / sum_{y=0} w - 1
                     (E_ref[r] = 1 for the true ratio; analysis gate |IC| < 0.01)
  shape closure      quantile-bin the score; per bin b compare the r_hat-reweighted
                     reference mass A_b = sum_{y=0,b} w r_hat / W0 with the actual
                     hypothesis mass B_b = sum_{y=1,b} w / W1, rel_b = A_b/B_b - 1,
                     with per-bin stat error sigma_b^2 = 1/n0_eff + 1/n1_eff. Reported:
                       shape_rms    sqrt(mean(rel^2 - sigma^2) clipped at 0) -- the
                                    stat-DEBIASED RMS shape bias, in fractional units;
                                    gate against CLOSURE_TOL=0.05
                       shape_chi2ndf  sum((rel/sigma)^2)/nbins -- ~1 when deviations are
                                    pure noise; >>1 flags real miscalibration
                       shape_max    raw max_b |rel_b| (reporting only -- its noise floor
                                    exceeds 0.05 at realistic test sizes; do NOT gate on it)

IC is also statistics-limited (sigma_IC ~ sqrt(Var_ref[r]/n0)); read the seed spread as
the noise floor and compare configs at FIXED test set (the sweep shares one).

Self-test (`--self-test`, no torch needed): an analytically calibrated two-Gaussian toy
must pass both gates; a temperature-miscalibrated copy (logits x 1.5) must fail both.

Layout, tags and score convention are shared with plot_data_efficiency.py:
  <store>/predictions/evenet-klambda-<config>-size<size>-seed<seed>/prediction.pt
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_data_efficiency import CONFIGS, _TAG, _softmax_signal  # noqa: E402

_EPS = 1e-6


PRIORS = ("trained", "balanced", "weighted", "counts")


def load_norm(path):
    """-> (class_balance, class_counts) as float arrays, from normalization.pt."""
    import torch
    d = torch.load(path, map_location="cpu")
    to_np = lambda v: np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v,
                                 dtype=float)
    return to_np(d["class_balance"]), to_np(d["class_counts"])


def _prior_factor(prior, y, w, norm=None):
    """Factor converting the learned odds into the density ratio.

    "trained" is the one implied by EveNet's own loss. That loss is
        sum_i c_{y_i} w_i CE_i / sum_i c_{y_i} w_i
    (evenet/network/loss/classification.py; c = normalization.pt["class_balance"],
    w = the SIGNED per-event weight, since Training.apply_event_weight is true), whose
    population optimum is  p/(1-p) = c1*dW1(x) / (c0*dW0(x)).  The physical ratio is
    r = [dW1/W1]/[dW0/W0], so

        r = p/(1-p) * (c0*W0) / (c1*W1),      W_c = class_counts[c]

    with W taken from normalization.pt, i.e. the SIGNED weighted class sums of the TRAIN
    split -- the same quantities the loss normalised by. Using test-split or |w| sums
    instead leaves a residual (for our data: 0.9493 vs 0.8593, a 10.5% overshoot).
    The other three options are the historical guesses, kept for comparison.
    """
    if prior == "trained":
        if norm is None:
            raise ValueError("prior='trained' needs --normalization <normalization.pt>")
        cb, cc = norm
        return float(cb[0] * cc[0]) / float(cb[1] * cc[1])
    if prior == "balanced":
        return 1.0
    if prior == "weighted":
        return w[~y].sum() / w[y].sum()
    if prior == "counts":
        return (~y).sum() / max(y.sum(), 1)
    raise ValueError(f"unknown prior {prior!r}")


def closure_metrics(p, y, w, nbins=20, neff_min=25.0, prior="balanced",
                    use_abs_w=True, norm=None, keep_bins=None):
    """-> dict(integral, integral_err, shape_rms, shape_rms_norm, shape_chi2ndf,
    shape_max, nbins_used, bins_kept). Pure numpy.

    keep_bins: freeze the accepted-bin set (indices) instead of re-deriving the
    neff_min acceptance from the current weights. Bootstrap replicas MUST pass the
    point estimate's bins_kept: Poisson(1) resampling doubles sum(w^2) per bin and so
    HALVES the Kish n_eff, silently dropping marginal bins from the replicas -- the
    spread would then belong to a different (fewer-bin, churning) statistic.

    integral        global normalisation error of r_hat (pre-registered gate)
    shape_rms       stat-debiased RMS bin deviation of r_hat as-is
    shape_rms_norm  same AFTER rescaling r_hat by 1/E_ref[r_hat] -- i.e. the SHAPE error
                    with the global normalisation divided out. Reported alongside (not
                    instead of) the pre-registered metrics: a global scale error is
                    degenerate with the rate term in an NSBI fit and removable by
                    construction, whereas a shape error is not. Post-hoc diagnostic,
                    added 2026-07-15 after the prior-convention fix.
    """
    nan = dict(integral=np.nan, integral_err=np.nan, shape_rms=np.nan,
               shape_rms_norm=np.nan, shape_chi2ndf=np.nan, shape_max=np.nan, nbins_used=0,
               bins_kept=[])
    p = np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)
    y = (np.asarray(y) > 0.5)
    w = np.asarray(w, float)
    if use_abs_w:                      # BCE training cannot consume negative weights;
        w = np.abs(w)                  # the reference density it saw is the |w| one
    W1, W0 = w[y].sum(), w[~y].sum()
    if W0 <= 0 or W1 <= 0:
        return nan
    r_hat = p / (1.0 - p) * _prior_factor(prior, y, w, norm)

    integral = float(np.sum(w[~y] * r_hat[~y]) / W0 - 1.0)
    res = w[~y] * (r_hat[~y] - (1.0 + integral))
    integral_err = float(np.sqrt(np.sum(res ** 2)) / W0)   # weighted SEM of E_ref[r]

    edges = np.quantile(p, np.linspace(0.0, 1.0, nbins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    edges = np.unique(edges)                     # ties in p can collapse bins
    idx = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)

    keep = set(keep_bins) if keep_bins is not None else None

    def bin_devs(r):
        rels, sigs, kept = [], [], []
        for b in range(len(edges) - 1):
            in_b = idx == b
            wb1 = w[y & in_b]                              # hypothesis side
            wb0r = w[~y & in_b] * r[~y & in_b]             # reweighted reference side
            n1 = wb1.sum() ** 2 / max((wb1 ** 2).sum(), _EPS)
            n0 = wb0r.sum() ** 2 / max((wb0r ** 2).sum(), _EPS)
            if (min(n1, n0) < neff_min) if keep is None else (b not in keep):
                continue                                   # ref bins at p->1 carry few, huge r
            kept.append(b)
            rels.append(wb0r.sum() / W0 / (wb1.sum() / W1) - 1.0)
            sigs.append(np.sqrt(1.0 / n0 + 1.0 / n1))
        return np.asarray(rels), np.asarray(sigs), kept

    def debiased_rms(rels, sigs):
        return float(np.sqrt(max(np.mean(rels ** 2 - sigs ** 2), 0.0)))

    rels, sigs, kept = bin_devs(r_hat)
    if not len(rels):
        return dict(nan, integral=integral, integral_err=integral_err)
    # shape with the global normalisation divided out (post-hoc decomposition)
    rels_n, sigs_n, _ = bin_devs(r_hat / (1.0 + integral))
    return dict(
        integral=integral,
        integral_err=integral_err,
        shape_rms=debiased_rms(rels, sigs),
        shape_rms_norm=(debiased_rms(rels_n, sigs_n) if len(rels_n) else np.nan),
        shape_chi2ndf=float(np.mean((rels / sigs) ** 2)),
        shape_max=float(np.abs(rels).max()),
        nbins_used=int(len(rels)),
        bins_kept=kept,
    )


def first_passing(results_cfg, int_gate=0.01, shape_gate=0.05):
    """Smallest fraction whose seed-mean passes BOTH gates (|IC| within max(gate, 3x stat),
    SC_rms below gate); None if no fraction passes. The closure left-shift of this number,
    finetune vs scratch, is the adoption metric."""
    for size in sorted(results_cfg):
        v = results_cfg[size]
        ic = np.nanmean(np.abs([m["integral"] for m in v]))
        ie = np.nanmean([m["integral_err"] for m in v])
        sc = np.nanmean([m["shape_rms"] for m in v])
        if ic <= max(int_gate, 3.0 * ie) and sc <= shape_gate:
            return size
    return None


def _load_prediction(path):
    import torch
    df = torch.load(path, map_location="cpu")
    logits = np.concatenate(
        [d["classification"]["classification/klambda"].numpy() for d in df], axis=0)
    label = np.concatenate([np.asarray(d["subprocess_id"]).reshape(-1) for d in df], axis=0)
    weight = np.concatenate([np.asarray(d["event_weight"]).reshape(-1) for d in df], axis=0)
    return _softmax_signal(logits), label, weight


def collect(store, nbins, neff_min, prior="balanced", use_abs_w=True, norm=None):
    """-> {config: {size: [metrics dict per seed]}}"""
    out = {c: {} for c in CONFIGS}
    for p in sorted(glob.glob(os.path.join(store, "predictions", "*", "prediction.pt"))):
        m = _TAG.search(os.path.basename(os.path.dirname(p)))
        if not m or m["cfg"] not in out:
            continue
        try:
            score, label, weight = _load_prediction(p)
            met = closure_metrics(score, label, weight, nbins, neff_min,
                                  prior=prior, use_abs_w=use_abs_w, norm=norm)
        except Exception as e:                    # noqa: BLE001
            print(f"  skip {p}: {e}")
            continue
        out[m["cfg"]].setdefault(float(m["size"]), []).append(met)
    return out


def plot(results, output, int_gate=0.01, shape_gate=0.05):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for ax, key, gate, title in ((axes[0], "integral", int_gate, "|integral closure|"),
                                 (axes[1], "shape_rms", shape_gate,
                                  "shape closure (stat-debiased RMS)")):
        for cfg in CONFIGS:
            d = results.get(cfg, {})
            if not d:
                continue
            xs = sorted(d)
            vals = [np.abs([m[key] for m in d[x]]) for x in xs]
            mean = np.array([np.nanmean(v) for v in vals])
            std = np.array([np.nanstd(v) for v in vals])
            ax.plot(xs, mean, marker="o", label=cfg)
            ax.fill_between(xs, np.clip(mean - std, 0, None), mean + std, alpha=0.2)
        ax.axhline(gate, ls="--", color="k", lw=1, label=f"gate {gate}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("training-set fraction")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("EveNet ratio-closure vs training fraction — HH→bbττ κ_λ")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"wrote {output}")


def self_test(n=200_000, mu=1.0, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) > 0.5).astype(int)
    x = rng.normal(loc=mu * y, scale=1.0)
    w = rng.uniform(0.5, 1.5, size=n)             # x-independent weights keep p calibrated
    logit = mu * x - 0.5 * mu ** 2                # exact log-ratio for equal priors
    p_cal = 1.0 / (1.0 + np.exp(-logit))
    p_bad = 1.0 / (1.0 + np.exp(-1.5 * logit))    # temperature-miscalibrated

    good = closure_metrics(p_cal, y, w)
    bad = closure_metrics(p_bad, y, w)
    for name, m in (("calibrated", good), ("miscalibrated", bad)):
        print(f"{name:13s} integral={m['integral']:+.4f}+-{m['integral_err']:.4f}  "
              f"shape_rms={m['shape_rms']:.4f}  chi2/ndf={m['shape_chi2ndf']:.2f}  "
              f"raw_max={m['shape_max']:.4f} (bins={m['nbins_used']})")
    assert abs(good["integral"]) < max(0.01, 3 * good["integral_err"]), \
        "calibrated toy failed the integral gate beyond its stat error"
    assert good["shape_rms"] < 0.05, "calibrated toy failed the shape gate"
    assert good["shape_chi2ndf"] < 3.0, "calibrated toy chi2/ndf inconsistent with noise"
    assert abs(bad["integral"]) > 0.01 and bad["shape_rms"] > 0.05, \
        "miscalibrated toy passed the gates -- metric is not sensitive"
    print("SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store_dir")
    ap.add_argument("--output-prefix", default="closure",
                    help="writes <prefix>.json and <prefix>.png")
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--neff-min", type=float, default=25.0)
    ap.add_argument("--prior", choices=PRIORS, default=None,
                    help="ratio prior convention; defaults to 'trained' when "
                         "--normalization is given, else 'balanced'")
    ap.add_argument("--normalization", help="evenet-train/normalization.pt -> enables "
                                            "prior='trained' (the convention EveNet's loss "
                                            "implies) and signed-weight closure")
    ap.add_argument("--signed-weights", action="store_true",
                    help="use stored signed weights instead of |w| (default |w|)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.store_dir:
        ap.error("--store_dir required unless --self-test")

    norm = load_norm(args.normalization) if args.normalization else None
    if args.prior is None:
        args.prior = "trained" if norm is not None else "balanced"
    # the trained convention is defined on the SIGNED weight densities the loss used
    use_abs = not args.signed_weights if args.prior != "trained" else False
    if args.prior == "trained":
        cb, cc = norm
        print(f"class_balance={cb}  class_counts={cc}")
        print(f"prior factor  = (c0*W0)/(c1*W1) = "
              f"{_prior_factor('trained', None, None, norm):.4f}")
    print(f"prior convention: {args.prior}   weights: {'|w|' if use_abs else 'signed'}")
    results = collect(args.store_dir, args.nbins, args.neff_min,
                      prior=args.prior, use_abs_w=use_abs, norm=norm)
    for cfg in CONFIGS:
        for size in sorted(results.get(cfg, {})):
            v = results[cfg][size]
            ic = np.abs([m["integral"] for m in v])
            ie = np.asarray([m["integral_err"] for m in v])
            sc = np.asarray([m["shape_rms"] for m in v])
            x2 = np.asarray([m["shape_chi2ndf"] for m in v])
            scn = np.asarray([m.get("shape_rms_norm", np.nan) for m in v])
            print(f"{cfg:9s} size={size:<5} |IC|={np.nanmean(ic):.4f}+-{np.nanstd(ic):.4f} "
                  f"(stat {np.nanmean(ie):.4f})  "
                  f"SC_rms={np.nanmean(sc):.4f}+-{np.nanstd(sc):.4f}  "
                  f"SC_norm={np.nanmean(scn):.4f}+-{np.nanstd(scn):.4f}  "
                  f"chi2/ndf={np.nanmean(x2):.2f}  (n={len(v)})")

    # Decision readout (primacy fixed 2026-07-12, pre-unblinding): the adoption metric is
    # the closure LEFT-SHIFT -- smallest fraction whose seed-mean passes both gates.
    # AUC is a secondary guard (finetune must not regress vs scratch), not the win metric.
    fp = {cfg: first_passing(results.get(cfg, {})) for cfg in CONFIGS}
    for cfg in CONFIGS:
        print(f"first fraction passing gates  {cfg:9s}: "
              f"{fp[cfg] if fp[cfg] is not None else 'none'}")
    if fp.get("finetune") and fp.get("scratch"):
        print(f"equivalent-data multiplier (scratch/finetune): "
              f"{fp['scratch'] / fp['finetune']:.1f}x")

    with open(f"{args.output_prefix}.json", "w") as f:
        json.dump(results, f, indent=1)
    plot(results, f"{args.output_prefix}.png")


if __name__ == "__main__":
    main()
