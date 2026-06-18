"""Merge per-shard partial clouds (<label>.part*.npz) into one <label>.npz per kappa_lambda
point, plus a manifest.json. Run after the convert_shard.py Slurm array finishes.

Memory note: merging loads all shards of ONE point at once, so peak memory ~ that point's
full cloud array. If a point is too big to merge in memory, train directly from the shards
instead (the dataset can take a list of shard files) and skip the merge.
"""
import argparse
import glob
import json
import os

import numpy as np

from convert_all_kl import KL_POINTS

_KEYS = ("parts", "part_vectors", "part_mask", "jet_mask", "obj", "obj_mask", "w")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keep-parts", action="store_true", help="don't delete the part*.npz after merge")
    args = ap.parse_args()

    manifest = {}
    for suf, (label, kl) in KL_POINTS.items():
        parts = sorted(glob.glob(os.path.join(args.out_dir, f"{label}.part*.npz")))
        if not parts:
            continue
        acc = {k: [] for k in _KEYS}
        for p in parts:
            with np.load(p) as d:
                for k in _KEYS:
                    acc[k].append(d[k])
        out = os.path.join(args.out_dir, f"{label}.npz")
        merged = {k: np.concatenate(acc[k], axis=0) for k in _KEYS}
        np.savez_compressed(out, **merged)
        n = int(merged["w"].shape[0])
        manifest[label] = {"kappa_lambda": kl, "n_shards": len(parts), "events": n, "npz": out}
        print(f"[merge] {label} (kl={kl}): {len(parts)} shard(s) -> {out} ({n} events)")
        if not args.keep_parts:
            for p in parts:
                os.remove(p)

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[done] merged {len(manifest)} point(s); manifest -> {args.out_dir}/manifest.json")


if __name__ == "__main__":
    main()
