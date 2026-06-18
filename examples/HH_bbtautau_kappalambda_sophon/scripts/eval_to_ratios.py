"""Constituent analog of data_nn_eval.py: build the Asimov event set, run ONNX
inference per basis process, convert score->ratio, aggregate ensembles, and save
ratio_<process>.npy + the Asimov weights.npy on the SHARED event ordering that the
JAX fit (sbi_parametric_model) loads."""
import numpy as np
from nsbi_common_utils.training.utils import predict_with_onnx_constituents, convert_score_to_ratio

_CLOUD_KEYS = ["parts", "part_mask", "jet_mask", "obj", "obj_mask"]


def build_asimov(processes, order):
    """Concatenate per-process clouds in a fixed order -> (asimov_clouds, weights)."""
    asimov = {k: np.concatenate([processes[p][k] for p in order], axis=0) for k in _CLOUD_KEYS}
    weights = np.concatenate([processes[p]["w"] for p in order], axis=0).astype(np.float64)
    return asimov, weights


def eval_process_ratio(asimov_clouds, onnx_paths, aggregation="mean_ratio", use_log_loss=False):
    """Evaluate one process's density ratio on the Asimov set, ensemble-aggregated.

    onnx_paths: str (single member) or list[str] (ensemble).
    """
    if isinstance(onnx_paths, str):
        onnx_paths = [onnx_paths]
    ratios = []
    for p in onnx_paths:
        score = predict_with_onnx_constituents(asimov_clouds, p)
        if use_log_loss:
            score = 1.0 / (1.0 + np.exp(-score))
        score = np.clip(score, 1e-9, 1.0 - 1e-9)
        ratios.append(convert_score_to_ratio(score))
    ratios = np.stack(ratios, axis=0)
    if aggregation == "median_ratio":
        return np.median(ratios, axis=0)
    return np.mean(ratios, axis=0)
