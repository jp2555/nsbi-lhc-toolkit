"""Money Plot: AUC vs training-set fraction, one curve per {finetune, frozen, scratch}.

Reads EveNet prediction.pt files (the format produced by EveNet-Full/scripts/predict.py and
consumed by Exotic-Higgs Produce_ntuple.py): a list of dicts with
  d["classification"]["classification/klambda"]  -> logits (..., 2)
  d["subprocess_id"]                              -> per-event class id (0=ref, 1=kl_hyp)
  d["event_weight"]                               -> per-event weight
Computes a weighted AUC per (config, dataset_size, seed), then plots mean +- std vs dataset_size.
Overlay the kinematic-ceiling AUC with --ceiling (a single float or a JSON {size: auc}).

The weighted-AUC core is pure numpy (unit-tested locally); torch/matplotlib are imported lazily
so this file can be checked without them. Expected layout:
  <store>/predictions/evenet-klambda-<config>-size<size>-seed<seed>/prediction.pt
"""
import argparse
import glob
import json
import os
import re

import numpy as np

CONFIGS = ["finetune", "frozen", "scratch"]
_TAG = re.compile(r"evenet-klambda-(?P<cfg>[a-zA-Z]+)-size(?P<size>[0-9.]+)-seed(?P<seed>[0-9]+)")


def weighted_auc(score, label, weight):
    """Trapezoidal weighted ROC-AUC with tie grouping. score: higher = more signal-like."""
    score = np.asarray(score, float)
    label = (np.asarray(label) > 0.5).astype(float)
    weight = np.asarray(weight, float)
    P = (weight * label).sum()
    N = (weight * (1.0 - label)).sum()
    if P <= 0 or N <= 0:
        return float("nan")
    order = np.argsort(-score)
    s, y, w = score[order], label[order], weight[order]
    tp = fp = prev_tpr = prev_fpr = auc = 0.0
    i, n = 0, len(s)
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        tp += (w[i:j] * y[i:j]).sum()
        fp += (w[i:j] * (1.0 - y[i:j])).sum()
        tpr, fpr = tp / P, fp / N
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_tpr, prev_fpr = tpr, fpr
        i = j
    return auc


def _softmax_signal(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=-1, keepdims=True))[..., 1]


def auc_from_prediction(path):
    import torch
    df = torch.load(path, map_location="cpu")
    logits = np.concatenate(
        [d["classification"]["classification/klambda"].numpy() for d in df], axis=0)
    label = np.concatenate([np.asarray(d["subprocess_id"]).reshape(-1) for d in df], axis=0)
    weight = np.concatenate([np.asarray(d["event_weight"]).reshape(-1) for d in df], axis=0)
    return weighted_auc(_softmax_signal(logits), label, weight)


def collect(store):
    """-> {config: {size: [auc per seed]}}"""
    out = {c: {} for c in CONFIGS}
    for p in glob.glob(os.path.join(store, "predictions", "*", "prediction.pt")):
        m = _TAG.search(os.path.basename(os.path.dirname(p)))
        if not m or m["cfg"] not in out:
            continue
        size = float(m["size"])
        try:
            a = auc_from_prediction(p)
        except Exception as e:                       # noqa: BLE001
            print(f"  skip {p}: {e}")
            continue
        out[m["cfg"]].setdefault(size, []).append(a)
    return out


def plot(curves, ceiling, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for cfg in CONFIGS:
        d = curves.get(cfg, {})
        if not d:
            continue
        xs = sorted(d)
        mean = np.array([np.nanmean(d[x]) for x in xs])
        std = np.array([np.nanstd(d[x]) for x in xs])
        ax.plot(xs, mean, marker="o", label=cfg)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.2)
    if ceiling is not None:
        if isinstance(ceiling, dict):
            xs = sorted(float(k) for k in ceiling)
            ax.plot(xs, [ceiling[str(x)] if str(x) in ceiling else ceiling[x] for x in xs],
                    "k--", label="kinematic ceiling")
        else:
            ax.axhline(ceiling, ls="--", color="k", label="kinematic ceiling")
    ax.set_xscale("log")
    ax.set_xlabel("training-set fraction")
    ax.set_ylabel("weighted AUC")
    ax.set_title("EveNet data-efficiency — HH→bbττ κ_λ")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"wrote {output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store_dir", required=True)
    ap.add_argument("--output", default="data_efficiency.png")
    ap.add_argument("--ceiling", default=None,
                    help="float, or path to JSON {size: auc} from run_kinematic_baseline.py")
    args = ap.parse_args()

    ceiling = None
    if args.ceiling:
        if os.path.exists(args.ceiling):
            ceiling = json.load(open(args.ceiling))
        else:
            ceiling = float(args.ceiling)

    curves = collect(args.store_dir)
    for cfg in CONFIGS:
        for size in sorted(curves.get(cfg, {})):
            v = curves[cfg][size]
            print(f"{cfg:9s} size={size:<5} AUC={np.nanmean(v):.4f}+-{np.nanstd(v):.4f} (n={len(v)})")
    plot(curves, ceiling, args.output)


if __name__ == "__main__":
    main()
