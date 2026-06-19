"""Systematics-robustness study: the FM's *actual* test (not BCE/AUC).

For each discriminant -- the high-level-feature BDT (the at-ceiling baseline) and,
optionally, a constituent/FM model -- we:
  1. apply analytic systematic variations to the SAME events (JES, tau-ES, parton
     shower), and measure how much the discriminant output MOVES (shape shift); a more
     robust discriminant moves less;
  2. fold those shifts into a simplified profiled-Asimov fit and report the uncertainty
     on a signal strength mu (a stand-in for the kappa_lambda CI) with vs without the
     systematics -- so we can ask whether a more robust discriminant gives a tighter
     *profiled* interval.

This is the experiment argued for in the design note (Sec. "separation vs systematics
robustness"): an FM can help the kappa_lambda *measurement* via robustness even though
it does not improve *separation*. Robustness is invisible to BCE/AUC -- it shows up here.

SCOPE / PROXIES (this is a Delphes-pheno scaffold, not the real CMS variations):
  * JES  : uniform jet energy scale. Relative constituent features (logptrel, logerel,
           angles) are invariant under a uniform scale, so only parts[:,17:21] (4-vec)
           and cols 0,1 (std log-pt/log-E) shift -- applied exactly.
  * tauES: leptonic-tau energy scale on e/mu object tokens. Hadronic-tau JETS cannot be
           isolated in the current clouds (no b/tau jet label) -> tau_h-jet ES is NOT
           applied (known limitation; add jet labels to enable).
  * shower: random soft-constituent dropout proxy (shower-model soft-radiation diff).
  * The profiled fit uses class-b as "signal", class-a as "background" with illustrative
    yields (--n-sig/--n-bkg), and mu as a kappa_lambda proxy. The real kappa_lambda CI
    needs the morphing fit (sbi_parametric_model) -- TODO hook.

Usage (feature-BDT only; CPU):
  pixi run -e nsbi-env python run_systematics_robustness.py \
      --clouds-dir $SCRATCH/dihiggs/clouds_subset --point-a kl5 --point-b kl0 \
      --num 20000 --out-dir syst_kl0_vs_kl5
Add the constituent/FM discriminant (GPU env; needs the checkpoint for sophon controls):
  ... --with-constituent --controls scratch sophon_finetune \
      --checkpoint $SCRATCH/sophon-ak4/models/JetClassII_SophonAK4/model.pt
"""
import argparse
import json
import os

import numpy as np

from run_kinematic_baseline import (
    load_point, _shuffle, build_binary_task, high_level_features,
)

_CLOUD_KEYS = ("parts", "part_mask", "jet_mask", "obj", "obj_mask")


# --------------------------- systematic variations ---------------------------
# Each takes a task/cloud dict and returns a modified COPY (same keys).

def _copy(d):
    return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in d.items()}


def vary_jes(d, delta):
    """Uniform jet energy scale by (1+delta). Scales constituent 4-vectors (cols 17:21)
    and shifts the standardized log-pt/log-E (cols 0,1) by 0.7*log(1+delta) [the sophon
    std 'multiply'], re-clipped to [-5,5]. Relative features are scale-invariant."""
    out = _copy(d)
    parts = d["parts"].copy()
    real = d["part_mask"] > 0
    s = 1.0 + delta
    vec = parts[..., 17:21] * s
    parts[..., 17:21] = np.where(real[..., None], vec, parts[..., 17:21])
    shift = 0.7 * np.log(s)
    for col in (0, 1):
        parts[..., col] = np.where(real, np.clip(parts[..., col] + shift, -5.0, 5.0),
                                   parts[..., col])
    out["parts"] = parts
    return out


def vary_taues(d, delta):
    """Leptonic-tau energy scale on e/mu object tokens (type 1,2). obj =
    [log_pt, eta, sin, cos, val, type_id]; scale pt -> log_pt += log(1+delta).
    tau_h-jet ES not applied (no jet flavour label in the current clouds)."""
    out = _copy(d)
    obj = d["obj"].copy()
    tid = obj[..., 5]
    lep = (d["obj_mask"] > 0) & ((tid == 1) | (tid == 2))
    obj[..., 0] = np.where(lep, obj[..., 0] + np.log(1.0 + delta), obj[..., 0])
    out["obj"] = obj
    return out


def vary_shower(d, drop_frac, seed=0):
    """Parton-shower proxy: drop each real constituent independently with probability
    drop_frac (emulates shower-model soft-radiation differences). Only part_mask changes;
    jet 4-vectors are re-summed downstream from the survivors. (A softness-weighted drop
    would be a refinement.)"""
    out = _copy(d)
    pm = d["part_mask"].copy()
    rng = np.random.default_rng(seed)
    keep = (rng.random(pm.shape) >= drop_frac).astype(pm.dtype)
    out["part_mask"] = pm * keep
    return out


def make_variations(jes, taues, shower, seed):
    """name -> (fn, systematic_group). Up/down share a group -> one nuisance each."""
    return {
        "JES_up":     (lambda d: vary_jes(d, +jes), "JES"),
        "JES_down":   (lambda d: vary_jes(d, -jes), "JES"),
        "TauES_up":   (lambda d: vary_taues(d, +taues), "TauES"),
        "TauES_down": (lambda d: vary_taues(d, -taues), "TauES"),
        "Shower":     (lambda d: vary_shower(d, shower, seed), "Shower"),
    }


# --------------------------- metrics ---------------------------

def _wmean(x, w):
    return float(np.average(x, weights=w))


def _hist(scores, w, bins):
    h, _ = np.histogram(scores, bins=bins, weights=w)
    return h


def _tvd(h1, h2):
    """total-variation distance between two (shape-normalized) histograms in [0,1]."""
    a = h1 / max(h1.sum(), 1e-12)
    b = h2 / max(h2.sum(), 1e-12)
    return float(0.5 * np.abs(a - b).sum())


def robustness_metrics(scores_nom, scores_var, w, bins):
    """How much the discriminant moves under one variation (same events -> paired)."""
    return {
        "rms_dscore": float(np.sqrt(_wmean((scores_var - scores_nom) ** 2, w))),
        "mean_shift": _wmean(scores_var, w) - _wmean(scores_nom, w),
        "shape_tvd": _tvd(_hist(scores_nom, w, bins), _hist(scores_var, w, bins)),
    }


def asimov_sigma_mu(scores, varied_scores, y, w, groups, n_sig, n_bkg, bins):
    """Simplified profiled-Asimov uncertainty on signal strength mu.

    Templates: class b (y==1) = 'signal' scaled to n_sig, class a (y==0) = 'background'
    scaled to n_bkg. Each systematic group contributes one Gaussian-constrained shape
    nuisance whose per-bin shift is the (symmetrized) change in the TOTAL template.
    Returns (sigma_mu_stat_only, sigma_mu_with_syst)."""
    def tot_template(sc):
        s = _hist(sc[y == 1], w[y == 1], bins)
        b = _hist(sc[y == 0], w[y == 0], bins)
        s = s / max(s.sum(), 1e-12) * n_sig
        b = b / max(b.sum(), 1e-12) * n_bkg
        return s, b

    s0, b0 = tot_template(scores)
    n = s0 + b0                                   # Asimov expectation at mu=1, theta=0
    n = np.clip(n, 1e-9, None)

    # per-group template shift (delta n / delta theta)
    deltas = []
    for g, names in groups.items():
        if len(names) == 2:                       # up/down -> symmetric half-difference
            up = sum(tot_template(varied_scores[names[0]]))
            dn = sum(tot_template(varied_scores[names[1]]))
            deltas.append(0.5 * (up - dn))
        else:                                     # one-sided -> shift vs nominal
            v = sum(tot_template(varied_scores[names[0]]))
            deltas.append(v - (s0 + b0))

    # Fisher information for (mu, theta_1..theta_K); Asimov + Gaussian nuisance constraints
    grads = [s0] + deltas                         # dn/dmu = s0 ; dn/dtheta_k = delta_k
    P = len(grads)
    info = np.zeros((P, P))
    for a in range(P):
        for c in range(P):
            info[a, c] = np.sum(grads[a] * grads[c] / n)
    for k in range(1, P):
        info[k, k] += 1.0                         # unit Gaussian constraint per nuisance
    sigma_syst = float(np.sqrt(np.linalg.inv(info)[0, 0]))
    sigma_stat = float(1.0 / np.sqrt(np.sum(s0 * s0 / n)))
    return sigma_stat, sigma_syst


# --------------------------- discriminants ---------------------------

def feature_scorer(task_nom):
    """Train the high-level-feature BDT on nominal; return a scorer fn over task dicts."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         early_stopping=True, validation_fraction=0.15,
                                         random_state=0)
    clf.fit(high_level_features(task_nom), task_nom["y"], sample_weight=task_nom["w"])
    return lambda task: np.clip(clf.predict_proba(high_level_features(task))[:, 1],
                                1e-9, 1.0 - 1e-9)


def constituent_scorer(task_nom, point_a, point_b, control, checkpoint, out_dir,
                       epochs, batch_size, learning_rate):
    """Train a constituent control on nominal; return a scorer fn over task dicts.
    Requires torch + the toolkit (GPU env). Reuses the diagnostic's trainer/controls."""
    import torch
    from run_kappa_diagnostic import _controls, _CONTROL_LR
    from nsbi_common_utils.training.particle_ratio_estimation import particle_density_ratio_trainer
    from nsbi_common_utils.lightning_tools.cloud_spec import SOPHON_SPEC

    kind, freeze, ekw = _controls(checkpoint)[control]
    lr = _CONTROL_LR.get(control, learning_rate)
    tr = particle_density_ratio_trainer(
        clouds=task_nom, sample_name=[point_b, point_a],
        output_name=f"syst_{control}", path_to_models=os.path.join(out_dir, control) + "/",
        encoder_kind=kind, spec=SOPHON_SPEC, freeze_backbone=freeze, encoder_kwargs=ekw)
    tr.train(number_of_epochs=epochs, batch_size=batch_size, learning_rate=lr,
             holdout_split=0.3, export_onnx=False)
    model = tr.model.eval()
    dev = next(model.parameters()).device

    def score(task):
        outs = []
        with torch.no_grad():
            for i in range(0, task["y"].shape[0], batch_size):
                sl = slice(i, i + batch_size)
                batch = {k: torch.as_tensor(np.asarray(task[k][sl])).to(dev) for k in _CLOUD_KEYS}
                outs.append(model(batch).detach().cpu().numpy().ravel())
        return np.clip(np.concatenate(outs), 1e-9, 1.0 - 1e-9)
    return score


# --------------------------- driver ---------------------------

def run(task, title, out_dir, jes, taues, shower, seed, n_sig, n_bkg, nbins,
        discriminants):
    os.makedirs(out_dir, exist_ok=True)
    variations = make_variations(jes, taues, shower, seed)
    groups = {}
    for name, (_, g) in variations.items():
        groups.setdefault(g, []).append(name)
    bins = np.linspace(0.0, 1.0, nbins + 1)

    results = {}
    for dname, scorer in discriminants.items():
        print(f"[syst] discriminant: {dname}")
        sc_nom = scorer(task)
        sc_var = {name: scorer(fn(task)) for name, (fn, _) in variations.items()}
        rob = {name: robustness_metrics(sc_nom, sc_var[name], task["w"], bins)
               for name in variations}
        s_stat, s_syst = asimov_sigma_mu(sc_nom, sc_var, task["y"], task["w"],
                                         groups, n_sig, n_bkg, bins)
        results[dname] = {
            "robustness": rob,
            "sigma_mu_stat": s_stat,
            "sigma_mu_syst": s_syst,
            "syst_inflation": s_syst / s_stat if s_stat else float("nan"),
        }
        for name in variations:
            print(f"    {name:10s}: shape_tvd={rob[name]['shape_tvd']:.4f}  "
                  f"rms_dscore={rob[name]['rms_dscore']:.4f}")
        print(f"    sigma_mu: stat={s_stat:.4f}  stat+syst={s_syst:.4f}  "
              f"inflation={results[dname]['syst_inflation']:.3f}")

    with open(os.path.join(out_dir, "robustness.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    _plot(results, list(variations), out_dir, title)
    return results


def _plot(results, var_names, out_dir, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                       # pragma: no cover
        print(f"[plot] skipped ({exc})")
        return
    discs = list(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(var_names))
    width = 0.8 / max(len(discs), 1)
    for i, d in enumerate(discs):
        tvds = [results[d]["robustness"][v]["shape_tvd"] for v in var_names]
        ax1.bar(x + i * width, tvds, width, label=d)
    ax1.set_xticks(x + width * (len(discs) - 1) / 2)
    ax1.set_xticklabels(var_names, rotation=30, ha="right")
    ax1.set_ylabel("discriminant shape shift (TVD)")
    ax1.set_title(f"robustness: {title}  (lower = more robust)")
    ax1.legend()

    xb = np.arange(len(discs))
    ax2.bar(xb - 0.2, [results[d]["sigma_mu_stat"] for d in discs], 0.4, label="stat only")
    ax2.bar(xb + 0.2, [results[d]["sigma_mu_syst"] for d in discs], 0.4, label="stat+syst")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(discs, rotation=20, ha="right")
    ax2.set_ylabel(r"$\sigma_\mu$ (Asimov, $\mu$ as $\kappa_\lambda$ proxy)")
    ax2.set_title("profiled uncertainty (lower = better)")
    ax2.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "robustness.png")
    fig.savefig(path, dpi=120)
    print(f"[plot] -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clouds-dir", required=True)
    ap.add_argument("--point-a", default="kl5", help="reference / 'background' label")
    ap.add_argument("--point-b", default="kl0", help="'signal' label")
    ap.add_argument("--num", type=int, default=20000, help="events per point (0=all)")
    ap.add_argument("--jes", type=float, default=0.05, help="JES fractional shift")
    ap.add_argument("--taues", type=float, default=0.03, help="leptonic tau-ES shift")
    ap.add_argument("--shower-drop", type=float, default=0.05,
                    help="shower proxy: per-constituent drop probability")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-sig", type=float, default=50.0, help="Asimov signal yield (proxy)")
    ap.add_argument("--n-bkg", type=float, default=5000.0, help="Asimov background yield (proxy)")
    ap.add_argument("--nbins", type=int, default=20, help="discriminant histogram bins")
    ap.add_argument("--out-dir", default="syst_out")
    ap.add_argument("--with-constituent", action="store_true",
                    help="also evaluate constituent/FM controls (needs torch + GPU env)")
    ap.add_argument("--controls", nargs="+", default=["scratch", "sophon_finetune"])
    ap.add_argument("--checkpoint", default=os.environ.get("SOPHON_AK4_CKPT", ""))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    args = ap.parse_args()

    # build the nominal task once so every discriminant trains on identical events
    a = _shuffle(load_point(os.path.join(args.clouds_dir, f"{args.point_a}.npz"), args.num))
    b = _shuffle(load_point(os.path.join(args.clouds_dir, f"{args.point_b}.npz"), args.num))
    task_nom = build_binary_task(a, b)

    discriminants = {"feature_bdt": feature_scorer(task_nom)}
    if args.with_constituent:
        for c in args.controls:
            if c.startswith("sophon") and not args.checkpoint:
                print(f"[skip] {c}: set --checkpoint / SOPHON_AK4_CKPT")
                continue
            discriminants[c] = constituent_scorer(
                task_nom, args.point_a, args.point_b, c, args.checkpoint, args.out_dir,
                args.epochs, args.batch_size, args.learning_rate)

    run(task_nom, f"{args.point_b} vs {args.point_a}", args.out_dir,
        args.jes, args.taues, args.shower_drop, args.seed, args.n_sig, args.n_bkg,
        args.nbins, discriminants)


if __name__ == "__main__":
    main()
