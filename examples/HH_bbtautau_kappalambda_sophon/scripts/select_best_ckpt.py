"""Fix the checkpoint EveNet predict will load: the BEST-val one, per run directory.

EveNet_Public's predict.py resolves a checkpoint DIRECTORY via
    max(glob("*.ckpt"), key=os.path.getmtime)          # evenet/predict.py:39
i.e. the most-recently-written file = the FINAL epoch. At low statistics the final
epoch is the overfit endpoint (train falls, val rises past the early-stop best), so
evaluating it would systematically degrade the calibration readout and contaminate
the arm comparison with checkpoint-selection noise.

Pre-registered selection rule (2026-07-14, applied identically to all arms):
every run is evaluated at its BEST val/loss checkpoint. Training filenames embed it
(epoch=E_train=T_val=V.ckpt), so this script parses V, and `touch`es the best file
so the mtime rule picks it. Dangling/stale `last.ckpt` symlinks are removed (a
dangling symlink would crash predict's getmtime; a live one would shadow the touch).

--prune additionally DELETES all non-best epoch checkpoints (315 MB each; a full
sweep otherwise accumulates hundreds of GB). Selection alone is non-destructive.

Usage:
  python select_best_ckpt.py --store $STORE            # touch best per run dir
  python select_best_ckpt.py --store $STORE --prune    # + delete non-best ckpts
"""
import argparse
import glob
import os
import re
import time

_VAL = re.compile(r"val=([0-9]+(?:\.[0-9]+)?)\.ckpt$")


def best_of(dirpath):
    """-> (best_path, [other ckpts], [symlinks]) or (None, [], []) if no parsable ckpt."""
    cands, others, links = [], [], []
    for p in glob.glob(os.path.join(dirpath, "*.ckpt")):
        if os.path.islink(p):
            links.append(p)
            continue
        m = _VAL.search(os.path.basename(p))
        if m:
            cands.append((float(m.group(1)), p))
        else:
            others.append(p)
    if not cands:
        return None, others, links
    cands.sort(key=lambda t: t[0])
    best = cands[0][1]
    rest = [p for _, p in cands[1:]] + others
    return best, rest, links


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="STORE dir (contains checkpoints/)")
    ap.add_argument("--prune", action="store_true", help="delete non-best epoch ckpts")
    args = ap.parse_args()

    root = os.path.join(args.store, "checkpoints")
    dirs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
    if not dirs:
        raise SystemExit(f"no run dirs under {root}")
    n_sel = n_pruned = 0
    for d in dirs:
        best, rest, links = best_of(d)
        if best is None:
            print(f"SKIP {os.path.basename(d)}: no epoch=*_val=*.ckpt files")
            continue
        for l in links:                       # last.ckpt would shadow (or crash) the mtime rule
            os.remove(l)
        os.utime(best, (time.time(), time.time()))
        n_sel += 1
        if args.prune:
            for p in rest:
                os.remove(p)
                n_pruned += 1
        print(f"{os.path.basename(d)}: -> {os.path.basename(best)}"
              + (f"  (pruned {len(rest)})" if args.prune else ""))
    print(f"selected best-val in {n_sel}/{len(dirs)} dirs"
          + (f", pruned {n_pruned} ckpts" if args.prune else ""))


if __name__ == "__main__":
    main()
