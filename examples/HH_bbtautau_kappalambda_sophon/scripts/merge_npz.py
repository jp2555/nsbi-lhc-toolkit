"""Merge EveNet NPZ files into ONE shuffled NPZ for preprocessing.

EveNet_Public's preprocess.py processes each --files entry independently and its
per-class accounting assumes every file contains ALL classification classes
(np.bincount without minlength -> IndexError on a single-class file). Our adapter
emits one single-class NPZ per sample, so the pair must be merged first.

The shuffle also matters physically: adapter output is ordered by input file
(mt/*.root then et/*.root), so a sequential train/val/test split of an unshuffled
file would give a test split dominated by the et channel. A fixed-seed permutation
makes every split channel- and class-mixed.

Usage:
  python merge_npz.py --inputs kl1.npz kl5.npz --output merged.npz [--seed 123]
"""
import argparse

import numpy as np


def merge(paths, seed=123):
    parts = [dict(np.load(p, allow_pickle=True)) for p in paths]
    keys = set(parts[0])
    for p, d in zip(paths, parts):
        if set(d) != keys:
            raise KeyError(f"{p} keys {sorted(set(d))} != {sorted(keys)}")
    out = {k: np.concatenate([d[k] for d in parts], axis=0) for k in keys}
    n = len(out["classification"])
    perm = np.random.default_rng(seed).permutation(n)
    out = {k: v[perm] for k, v in out.items()}
    counts = np.bincount(out["classification"].astype(np.int64))
    print(f"merged {len(paths)} files -> {n} events, class counts {counts.tolist()}, "
          f"shuffled (seed {seed})")
    if (counts > 0).sum() < 2:
        raise ValueError("merged file still single-class — wrong input pair?")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    out = merge(args.inputs, args.seed)
    np.savez_compressed(args.output, **out)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
