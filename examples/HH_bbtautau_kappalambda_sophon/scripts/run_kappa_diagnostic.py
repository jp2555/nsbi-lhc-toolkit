"""Phase-1 diagnostic: does sophon-ak4 pretraining help the density-ratio classifier?

Binary task = (numerator point) vs the **kappa_lambda=5 reference (denominator)** — a
signal-shape difference, no background needed. Default = kl0 vs kl5 (box vs triangle-
dominated -> well separated). Trains the controls and writes loss curves + results.json:

  scratch          : ParT (sophon-ak4 shape) random init      -> isolates the pretraining benefit
  sophon_frozen    : sophon-ak4 backbone frozen, head trained  -> foundation, frozen
  sophon_finetune  : sophon-ak4 backbone fine-tuned end-to-end -> foundation, fine-tuned

Usage:
  export SOPHON_AK4_CKPT=/path/to/model.pt
  python run_kappa_diagnostic.py --clouds-dir $SCRATCH/dihiggs/clouds_subset \
      --point-a kl0 --point-b kl1 --num 20000 --epochs 30 --out-dir diagnostic_out

Notes:
  * --num caps events PER POINT (np.load reads the file fully, then slices), so point it at
    the subset clouds, not the full 45 GB/point files.
  * scratch/sophon controls need the vendored ParT (_part_vendor/) present; sophon controls
    additionally need SOPHON_AK4_CKPT (or --checkpoint). Missing prerequisites are skipped.
"""
import argparse
import json
import os

import numpy as np

from nsbi_common_utils.training.particle_ratio_estimation import particle_density_ratio_trainer
from nsbi_common_utils.lightning_tools.cloud_spec import SOPHON_SPEC

_CLOUD_KEYS = ("parts", "part_mask", "jet_mask", "obj", "obj_mask")


def _controls(checkpoint):
    """control -> (encoder_kind, freeze_backbone, extra encoder_kwargs)."""
    return {
        # same arch+inputs as sophon (17 feats + 4-vec pairwise), random init:
        "scratch":         ("scratch",    False, {"input_dim": 17, "use_pair": True}),
        "sophon_frozen":   ("sophon-ak4", True,  {"checkpoint": checkpoint}),
        "sophon_finetune": ("sophon-ak4", False, {"checkpoint": checkpoint}),
    }


# per-control learning-rate override (falls back to the global --learning-rate).
# fine-tuning a pretrained backbone overfits fast -> use a smaller LR than scratch/frozen.
_CONTROL_LR = {"sophon_finetune": 1e-4}


def load_point(npz_path, num=0):
    """Load one point's clouds (optionally capped to `num` events)."""
    d = np.load(npz_path)
    n = d["w"].shape[0]
    if num:
        n = min(num, n)
    return {k: d[k][:n] for k in (_CLOUD_KEYS + ("w",))}


def build_binary_task(clouds_a, clouds_b):
    """class 0 = a (denominator), class 1 = b (numerator); weights normalized per class."""
    out = {k: np.concatenate([clouds_a[k], clouds_b[k]], axis=0) for k in _CLOUD_KEYS}
    na, nb = clouds_a["w"].shape[0], clouds_b["w"].shape[0]
    wa, wb = np.abs(clouds_a["w"]).astype(np.float64), np.abs(clouds_b["w"]).astype(np.float64)
    wa = wa / wa.sum() if wa.sum() else wa
    wb = wb / wb.sum() if wb.sum() else wb
    out["y"] = np.concatenate([np.zeros(na), np.ones(nb)]).astype(np.float32)
    out["w"] = np.concatenate([wa, wb]).astype(np.float32)
    return out


def run(clouds_dir, point_a, point_b, num, checkpoint, out_dir, epochs, batch_size,
        controls, learning_rate=2e-4):
    os.makedirs(out_dir, exist_ok=True)
    a = load_point(os.path.join(clouds_dir, f"{point_a}.npz"), num)
    b = load_point(os.path.join(clouds_dir, f"{point_b}.npz"), num)
    task = build_binary_task(a, b)
    print(f"task {point_b}(1) vs {point_a}(0): {task['y'].shape[0]} events "
          f"({int(task['y'].sum())} signal)")

    cmap = _controls(checkpoint)
    results = {}
    for c in controls:
        kind, freeze, ekw = cmap[c]
        if kind == "sophon-ak4" and not checkpoint:
            print(f"[skip] {c}: set SOPHON_AK4_CKPT or --checkpoint")
            continue
        lr = _CONTROL_LR.get(c, learning_rate)
        print(f"[train] {c} (encoder={kind}, freeze={freeze}, lr={lr:g})")
        tr = particle_density_ratio_trainer(
            clouds=task, sample_name=[point_b, point_a],
            output_name=f"{point_b}_vs_{point_a}_{c}",
            path_to_models=os.path.join(out_dir, c) + "/",
            encoder_kind=kind, spec=SOPHON_SPEC, freeze_backbone=freeze, encoder_kwargs=ekw)
        hist = tr.train(number_of_epochs=epochs, batch_size=batch_size,
                        learning_rate=lr, holdout_split=0.3, export_onnx=False)
        results[c] = hist
        vls = hist.get("val_loss") or []
        best = min(vls) if vls else float("nan")
        best_ep = (int(np.argmin(vls)) + 1) if vls else -1
        last = vls[-1] if vls else float("nan")
        print(f"[done] {c}: best val_loss={best:.4f} @ epoch {best_ep} | last={last:.4f}")

    summary = {c: (min(h["val_loss"]) if h.get("val_loss") else float("nan"))
               for c, h in results.items()}
    if summary:
        print("[summary] best (min) val_loss: "
              + ", ".join(f"{c}={v:.4f}" for c, v in summary.items()))
        print(f"[summary] lowest best-val -> {min(summary, key=summary.get)} "
              "(lower = better generalization; the question is sophon_* vs scratch)")
    results["_best_val"] = summary
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    _plot(results, out_dir, f"{point_b} vs {point_a}")
    return results


def _plot(results, out_dir, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                       # pragma: no cover
        print(f"[plot] skipped ({exc})")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for c, h in results.items():
        vl = h.get("val_loss") or []
        if vl:
            ax.plot(range(1, len(vl) + 1), vl, marker="o", label=c)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation weighted-BCE loss")
    ax.set_title(f"density-ratio convergence: {title}")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "convergence.png")
    fig.savefig(path, dpi=120)
    print(f"[plot] -> {path}")


def _shuffle(d, seed=0):
    n = d["w"].shape[0]
    idx = np.random.default_rng(seed).permutation(n)
    return {k: v[idx] for k, v in d.items()}


def ablation(clouds_dir, point_a, point_b, sizes, checkpoint, out_dir, epochs,
             batch_size, controls, learning_rate=2e-4):
    """Train each control at several per-point event counts -> best val-loss vs N.

    This is the foundation-model data-efficiency test: the pretrained controls should
    beat `scratch` most at small N (and reach below the ln2 random floor sooner)."""
    os.makedirs(out_dir, exist_ok=True)
    nmax = max(sizes)
    a_full = _shuffle(load_point(os.path.join(clouds_dir, f"{point_a}.npz"), nmax))
    b_full = _shuffle(load_point(os.path.join(clouds_dir, f"{point_b}.npz"), nmax))
    cmap = _controls(checkpoint)
    curve = {c: {} for c in controls}
    for N in sorted(sizes):
        a = {k: v[:N] for k, v in a_full.items()}
        b = {k: v[:N] for k, v in b_full.items()}
        task = build_binary_task(a, b)
        print(f"[ablation] N={N}/point -> {task['y'].shape[0]} events")
        for c in controls:
            kind, freeze, ekw = cmap[c]
            if kind == "sophon-ak4" and not checkpoint:
                print(f"  [skip] {c}: no checkpoint")
                continue
            lr = _CONTROL_LR.get(c, learning_rate)
            tr = particle_density_ratio_trainer(
                clouds=task, sample_name=[point_b, point_a],
                output_name=f"{point_b}_vs_{point_a}_{c}_N{N}",
                path_to_models=os.path.join(out_dir, f"{c}_N{N}") + "/",
                encoder_kind=kind, spec=SOPHON_SPEC, freeze_backbone=freeze, encoder_kwargs=ekw)
            hist = tr.train(number_of_epochs=epochs, batch_size=batch_size,
                            learning_rate=lr, holdout_split=0.3, export_onnx=False)
            best = min(hist["val_loss"]) if hist.get("val_loss") else float("nan")
            curve[c][N] = best
            print(f"    {c}: best val_loss={best:.4f}")
    with open(os.path.join(out_dir, "ablation.json"), "w") as fh:
        json.dump(curve, fh, indent=2)
    _plot_ablation(curve, sorted(sizes), out_dir, f"{point_b} vs {point_a}")
    return curve


def _plot_ablation(curve, sizes, out_dir, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                       # pragma: no cover
        print(f"[plot] skipped ({exc})")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for c, d in curve.items():
        xs = [N for N in sizes if N in d]
        if xs:
            ax.plot(xs, [d[N] for N in xs], marker="o", label=c)
    ax.axhline(0.6931, ls="--", color="gray", lw=1, label="random (ln 2)")
    ax.set_xscale("log")
    ax.set_xlabel("events per point (N)")
    ax.set_ylabel("best validation weighted-BCE loss")
    ax.set_title(f"data-efficiency: {title}")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "ablation.png")
    fig.savefig(path, dpi=120)
    print(f"[plot] -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clouds-dir", required=True)
    ap.add_argument("--point-a", default="kl5", help="denominator / REFERENCE point label")
    ap.add_argument("--point-b", default="kl0", help="numerator point label")
    ap.add_argument("--num", type=int, default=100000, help="max events per point (0=all)")
    ap.add_argument("--ablation-sizes", nargs="*", type=int, default=None,
                    help="if set (e.g. 20000 50000 100000), run the data-efficiency ablation "
                         "instead of a single fixed-N run")
    ap.add_argument("--checkpoint", default=os.environ.get("SOPHON_AK4_CKPT", ""))
    ap.add_argument("--out-dir", default="diagnostic_out")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=2e-4,
                    help="LR for all controls; fine-tuning a pretrained backbone wants ~1e-4-5e-4")
    ap.add_argument("--controls", nargs="+",
                    default=["scratch", "sophon_frozen", "sophon_finetune"])
    args = ap.parse_args()
    if args.ablation_sizes:
        ablation(args.clouds_dir, args.point_a, args.point_b, args.ablation_sizes,
                 args.checkpoint, args.out_dir, args.epochs, args.batch_size,
                 args.controls, learning_rate=args.learning_rate)
    else:
        run(args.clouds_dir, args.point_a, args.point_b, args.num, args.checkpoint,
            args.out_dir, args.epochs, args.batch_size, args.controls,
            learning_rate=args.learning_rate)


if __name__ == "__main__":
    main()
