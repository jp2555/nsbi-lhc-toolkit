"""FAIR Universe systematics-robustness diagnostic (methodology testbed).

Measures how much a trained discriminant's output MOVES under the REAL FAIR Universe
systematic variations (TES, JES -- the simulator's own nuisances, not analytic proxies),
on H->tautau signal vs Z->tautau / ttbar backgrounds. Robustness is the axis that BCE/AUC
cannot see (see docs/.../fm-precision-classification note): a discriminant can separate
well yet move a lot under systematics, which is what limits a profiled measurement.

This complements parameter_fitting.py, which (with the CI reporting added alongside this
script) reports the profiled mu CI with vs without systematics -- the downstream impact.

Pipeline position: run AFTER neural_likelihood_ratio_estimation.py has trained the
nominal density-ratio ensemble for the chosen process. Reuses data_nn_eval.py's exact
config access, data loading, region filtering, and ensemble scoring.

For each region in {Nominal, JES_Up, JES_Dn, TES_Up, TES_Dn} present in the fit config:
  - load signal-region events (same physical events with systematically shifted features),
  - score the merged basis samples with the NOMINAL discriminant ensemble (median),
  - build the weighted score histogram,
  - measure the shift vs Nominal (total-variation distance, mean shift).
Writes robustness.json + robustness.png. A more robust discriminant moves less.

Usage:
  python scripts/systematics_robustness.py --config config.pipeline.yaml \
      --process htautau --out-dir output/robustness
"""
import argparse
import json
import os

import numpy as np
import yaml

import nsbi_common_utils

_REGIONS = ["Nominal", "JES_Up", "JES_Dn", "TES_Up", "TES_Dn"]


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def ensemble_scores(df_features, trained_models_path, process, n_members):
    """Median discriminant score over the trained ensemble members (mirrors data_nn_eval)."""
    preds = []
    for idx in range(n_members):
        d = os.path.join(trained_models_path, f"output_model_params_{process}{idx}/")
        onnx = f"{d}model{idx}.onnx"
        scaler = f"{d}model_scaler{idx}.bin"
        if not (os.path.exists(onnx) and os.path.exists(scaler)):
            continue
        sc, mdl = nsbi_common_utils.training.load_trained_model(onnx, scaler)
        preds.append(np.asarray(
            nsbi_common_utils.training.predict_with_model(df_features, sc, mdl)).ravel())
    if not preds:
        raise FileNotFoundError(
            f"no ensemble members for process '{process}' under {trained_models_path}")
    return np.median(np.stack(preds), axis=0)


def _tvd(h1, h2):
    a = h1 / max(h1.sum(), 1e-12)
    b = h2 / max(h2.sum(), 1e-12)
    return float(0.5 * np.abs(a - b).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="config.pipeline.yaml")
    ap.add_argument("--process", default=None,
                    help="which trained discriminant to probe (default: first basis process)")
    ap.add_argument("--n-members", type=int, default=None,
                    help="ensemble members (default: num_ensemble_members_evaluation from config)")
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--out-dir", default="output/robustness")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- config access mirrors data_nn_eval.py ---
    cfg = load_config(args.config)["neural_likelihood_ratio_estimation"]
    fit_cfg_path = cfg["nsbi_fit_config"]
    fit_cfg = nsbi_common_utils.configuration.ConfigManager(file_path_string=fit_cfg_path)
    features, _ = fit_cfg.get_training_features()
    basis = fit_cfg.get_basis_samples()
    process = args.process or basis[0]
    region = cfg["filter_region"]
    n_members = args.n_members or cfg.get("num_ensemble_members_evaluation", 1)
    trained_models_path = os.path.join(cfg["saved_data_path"].rstrip("/"), cfg["output_training_dir"])
    if not trained_models_path.endswith("/"):
        trained_models_path += "/"

    ds = nsbi_common_utils.datasets.datasets(
        config_path=fit_cfg_path, branches_to_load=features + ["presel_score"])
    all_regions = ds.load_datasets_from_config(load_systematics=True)
    regions = [r for r in _REGIONS if r in all_regions]
    print(f"[robustness] process={process}  members={n_members}  regions={regions}")

    # score every region's merged SR events with the nominal `process` discriminant
    scores, weights = {}, {}
    for r in regions:
        sr = ds.filter_region_dataset(all_regions[r].copy(), region=region)
        merged = ds.merge_dataframe_dict_for_training(sr, None, samples_to_merge=basis)
        scores[r] = ensemble_scores(merged[features], trained_models_path, process, n_members)
        weights[r] = merged["weights"].to_numpy()

    # shared bins from the nominal score range; shift metrics vs Nominal
    lo, hi = float(np.min(scores["Nominal"])), float(np.max(scores["Nominal"]))
    bins = np.linspace(lo, hi, args.nbins + 1)
    h_nom, _ = np.histogram(scores["Nominal"], bins=bins, weights=weights["Nominal"])
    mean_nom = float(np.average(scores["Nominal"], weights=weights["Nominal"]))

    out = {"process": process, "n_members": n_members, "regions": {}}
    for r in regions:
        h, _ = np.histogram(scores[r], bins=bins, weights=weights[r])
        out["regions"][r] = {
            "shape_tvd": _tvd(h_nom, h) if r != "Nominal" else 0.0,
            "mean_shift": float(np.average(scores[r], weights=weights[r])) - mean_nom,
            "n_events": int(scores[r].shape[0]),
        }
        print(f"    {r:9s}: shape_tvd={out['regions'][r]['shape_tvd']:.4f}  "
              f"mean_shift={out['regions'][r]['mean_shift']:+.4f}")

    with open(os.path.join(args.out_dir, "robustness.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    _plot(out, args.out_dir, process)


def _plot(out, out_dir, process):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                       # pragma: no cover
        print(f"[plot] skipped ({exc})")
        return
    syst = [r for r in out["regions"] if r != "Nominal"]
    tvds = [out["regions"][r]["shape_tvd"] for r in syst]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(syst, tvds, color="tab:purple")
    ax.set_ylabel("discriminant shape shift (TVD vs Nominal)")
    ax.set_title(f"systematics robustness: {process} discriminant\n(lower = more robust)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = os.path.join(out_dir, "robustness.png")
    fig.savefig(path, dpi=120)
    print(f"[plot] -> {path}")


if __name__ == "__main__":
    main()
