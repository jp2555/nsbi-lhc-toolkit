"""Convert ONE (kappa_lambda point, file-shard) -> a partial .npz, for Slurm-array parallelism.

A Slurm array task id ``t`` with ``--nshards S`` maps to:
    point  = list(KL_POINTS)[t // S]
    shard  = t % S
The point's files are round-robin sharded (``files[shard::S]``), so each of the S tasks for a
point converts ~1/S of its files. Output:
    <out_dir>/<label>.part<shard:03d>.npz
Merge the parts afterwards with ``merge_clouds.py``.

Total array size = len(KL_POINTS) * S  (= 6 * S). Tasks for absent points / empty shards no-op.
"""
import argparse
import glob
import os

from delphes_to_sophon_clouds import convert_tree
from convert_all_kl import KL_POINTS, _DSET, _FILE_GLOB

_ITEMS = list(KL_POINTS.items())   # stable order


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task-id", type=int, required=True, help="SLURM_ARRAY_TASK_ID")
    ap.add_argument("--nshards", type=int, default=20, help="file-shards per kappa_lambda point")
    ap.add_argument("--max-files", type=int, default=0, help="cap files per point (0=all; for quick tests)")
    ap.add_argument("--tree", default="Delphes")
    ap.add_argument("--step-size", type=int, default=20000)
    args = ap.parse_args()

    if args.task_id < 0 or args.task_id >= len(_ITEMS) * args.nshards:
        print(f"[noop] task {args.task_id} outside 0..{len(_ITEMS)*args.nshards - 1}")
        return
    pidx, shard = divmod(args.task_id, args.nshards)
    suf, (label, kl) = _ITEMS[pidx]

    pattern = os.path.join(args.raw_dir, _DSET.format(suf=suf), _FILE_GLOB)
    files = sorted(glob.glob(pattern))
    if args.max_files:
        files = files[: args.max_files]
    files = files[shard::args.nshards]
    if not files:
        print(f"[noop] {label} (kl={kl}) shard {shard}/{args.nshards}: no files")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{label}.part{shard:03d}.npz")
    print(f"[convert] {label} (kl={kl}) shard {shard}/{args.nshards}: {len(files)} file(s) -> {out}")
    convert_tree(files, args.tree, out, step_size=args.step_size)


if __name__ == "__main__":
    main()
