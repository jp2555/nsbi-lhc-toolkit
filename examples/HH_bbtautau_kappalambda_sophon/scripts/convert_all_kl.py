"""Drive delphes_to_sophon_clouds over the per-kappa_lambda HH->bbtautau Delphes datasets.

Expected layout on Perlmutter (as copied from GridKa dCache):

  <raw_dir>/GluGluHHto2B2Tau_Par-c2-0p00-kl-<SUF>-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8_Delphes/
           delphes-tree-*/delphes-tree_*.root

Produces one ``<out_dir>/<label>.npz`` per kappa_lambda point (sophon-ak4 clouds) plus a
``manifest.json`` mapping label -> {kappa_lambda, n_files, npz}. Points with no files are
skipped, so the same command works whether or not you copied all six.

Usage:
  python convert_all_kl.py --raw-dir $SCRATCH/dihiggs/raw --out-dir $SCRATCH/dihiggs/clouds
"""
import argparse
import glob
import json
import os

from delphes_to_sophon_clouds import convert_tree

# dataset-dir kl suffix -> (output label, kappa_lambda value)
KL_POINTS = {
    "0p00":  ("kl0",    0.0),
    "1p00":  ("kl1",    1.0),
    "2p45":  ("kl2p45", 2.45),
    "3p00":  ("kl3",    3.0),
    "5p00":  ("kl5",    5.0),
    "m1p00": ("klm1",  -1.0),
}
_DSET = "GluGluHHto2B2Tau_Par-c2-0p00-kl-{suf}-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8_Delphes"
_FILE_GLOB = "delphes-tree-*/delphes-tree_*.root"   # the hash subdir varies per dataset


def convert_all(raw_dir, out_dir, tree="Delphes", step_size=20000, max_files=0):
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}
    for suf, (label, kl) in KL_POINTS.items():
        pattern = os.path.join(raw_dir, _DSET.format(suf=suf), _FILE_GLOB)
        files = sorted(glob.glob(pattern))
        if max_files:
            files = files[:max_files]
        if not files:
            print(f"[skip] {label} (kl={kl}): no files match {pattern}")
            continue
        out = os.path.join(out_dir, f"{label}.npz")
        print(f"[convert] {label} (kl={kl}): {len(files)} file(s) -> {out}")
        convert_tree(files, tree, out, step_size=step_size)
        manifest[label] = {"kappa_lambda": kl, "n_files": len(files), "npz": out}
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[done] converted {len(manifest)} point(s); manifest -> {manifest_path}")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="dihiggs/raw",
                    help="dir containing the per-kappa_lambda GluGluHHto2B2Tau_* dataset dirs")
    ap.add_argument("--out-dir", default="dihiggs/clouds")
    ap.add_argument("--tree", default="Delphes")
    ap.add_argument("--step-size", type=int, default=20000)
    ap.add_argument("--max-files", type=int, default=0, help="cap files per point (0=all; for quick subset tests)")
    args = ap.parse_args()
    convert_all(args.raw_dir, args.out_dir, tree=args.tree,
                step_size=args.step_size, max_files=args.max_files)


if __name__ == "__main__":
    main()
