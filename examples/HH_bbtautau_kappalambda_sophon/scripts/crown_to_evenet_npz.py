"""CROWN bbtautau analysis ntuple (mt/et channels) -> EveNet NPZ (same contract as the
delphes/nanoaod adapters).

Input: the SAME files NSBI-pheno/dihiggs_bbtautau/convert_powheg_to_sbi.py reads -- flat
'ntuple' TTrees from the CROWN production, e.g.
  <base>/GluGluHHto2B2Tau_Par-c2-0p00-kl-{0p00,1p00,2p45,5p00}-kt-1p00_*PowhegBugFix*/{mt,et}/*.root
and the ttbar equivalents (convert_ttbar_to_sbi.py uses the identical branch contract).

These are post-selection events in EXACTLY the published phase space (the prelim-result
feature ntuple is derived from them), so the EveNet sweep and the feature ceiling are
apples-to-apples. Token cloud per event (4 objects):

  b1, b2  from bpair_{pt,eta,phi,mass}_{1,2}      btag=1 (the selected b-pair), charge 0
  tau_h   leg 2 ({pt,eta,phi,mass}_2)             hadronic group; --tau-encoding
                                                  anonymous=(0,0,0) | corner=(0,0,q_2)
  lep     leg 1 (mu in mt / e in et)              isLepton=1, charge q_1
  MET     met/metphi if present -> conditions     else 0 + loud warning
  weights genWeight * puweight (published convention; NLO signs preserved)

Optional branches resolved from candidates (first present wins), reported per file set:
  charge: q_1/q_2 (fallback: lep +1, tau -1 -- constant, documented, keeps the corner
          encoding off the light-jet corner but discards the real charge)
  met:    met/pfmet, metphi/pfmetphi

Events with non-finite or non-positive kinematics in any used branch are dropped and
counted (CROWN uses sentinel defaults when a candidate pair is absent).

Smoke mode reuses the shared fabricated-event generator (schema check only).
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delphes_to_evenet_npz import _finalize, N_OBJ_MAX, make_smoke, tau_corner_charge  # noqa: E402

_KIN = ["bpair_pt_1", "bpair_eta_1", "bpair_phi_1", "bpair_mass_1",
        "bpair_pt_2", "bpair_eta_2", "bpair_phi_2", "bpair_mass_2",
        "pt_1", "eta_1", "phi_1", "mass_1",
        "pt_2", "eta_2", "phi_2", "mass_2"]
_CAND = {
    "q_1": ["q_1", "charge_1"], "q_2": ["q_2", "charge_2"],
    "met": ["met", "pfmet"], "metphi": ["metphi", "pfmetphi"],
    "genWeight": ["genWeight"], "puweight": ["puweight"],
}
INPUT_TREE = "ntuple"


def _energy(pt, eta, mass):
    return np.sqrt((pt * np.cosh(eta)) ** 2 + np.clip(mass, 0.0, None) ** 2)


def load_files(patterns, max_events=0):
    import uproot
    paths = sorted(p for pat in patterns for p in glob.glob(pat))
    if not paths:
        raise FileNotFoundError(f"no files match {patterns}")
    print(f"reading {len(paths)} files (tree '{INPUT_TREE}')")
    chunks, resolved = [], None
    for path in paths:
        with uproot.open(f"{path}:{INPUT_TREE}") as t:
            have = set(t.keys())
            missing = [b for b in _KIN if b not in have]
            if missing:
                raise KeyError(f"{os.path.basename(path)} missing kinematics {missing}")
            if resolved is None:
                resolved = {k: next((c for c in cands if c in have), None)
                            for k, cands in _CAND.items()}
                print("optional branches:",
                      {k: (v or "ABSENT") for k, v in resolved.items()})
            chunks.append(t.arrays(_KIN + [v for v in resolved.values() if v],
                                   library="np"))
    out = {b: np.concatenate([c[b] for c in chunks]) for b in chunks[0]}
    if max_events:
        out = {k: v[:max_events] for k, v in out.items()}
    return out, resolved


def convert(patterns, class_id, tau_encoding, max_events=0):
    raw, res = load_files(patterns, max_events)
    n = len(raw["pt_1"])

    kin = np.stack([raw[b] for b in _KIN], axis=1)
    ok = np.isfinite(kin).all(axis=1)
    for pt_b in ("bpair_pt_1", "bpair_pt_2", "pt_1", "pt_2"):
        ok &= raw[pt_b] > 0
    n_drop = int(n - ok.sum())
    if n_drop:
        print(f"dropped {n_drop}/{n} events ({100 * n_drop / n:.1f}%) with "
              "non-finite/sentinel kinematics")
    raw = {k: v[ok] for k, v in raw.items()}
    m = len(raw["pt_1"])
    if not m:
        raise RuntimeError("no events survived the finiteness filter")

    q1 = raw[res["q_1"]].astype(np.float64) if res["q_1"] else np.full(m, 1.0)
    q2 = raw[res["q_2"]].astype(np.float64) if res["q_2"] else np.full(m, -1.0)
    if not res["q_1"]:
        print("WARNING: no charge branches -> lep q=+1, tau q=-1 constants "
              "(corner encoding stays valid but loses the real charge)")

    # hadronic group: b1, b2, tau_h -- pt-sorted per event; lepton last
    def tok(pt, eta, phi, mass, btag, islep, q):
        return np.stack([_energy(pt, eta, mass), pt, eta, phi,
                         np.full(m, btag), np.full(m, islep), q], axis=1)

    tau_q = np.where(q2 >= 0, 1.0, -1.0) if tau_encoding == "corner" else np.zeros(m)
    assert tau_corner_charge(1.0) == 1.0                    # shared convention guard
    had = np.stack([
        tok(raw["bpair_pt_1"], raw["bpair_eta_1"], raw["bpair_phi_1"], raw["bpair_mass_1"],
            1.0, 0.0, np.zeros(m)),
        tok(raw["bpair_pt_2"], raw["bpair_eta_2"], raw["bpair_phi_2"], raw["bpair_mass_2"],
            1.0, 0.0, np.zeros(m)),
        tok(raw["pt_2"], raw["eta_2"], raw["phi_2"], raw["mass_2"],
            0.0, 0.0, tau_q),
    ], axis=1)                                              # (m, 3, 7)
    order = np.argsort(-had[:, :, 1], axis=1)               # pt-desc within hadronic group
    had = np.take_along_axis(had, order[:, :, None], axis=1)
    lep = tok(raw["pt_1"], raw["eta_1"], raw["phi_1"], raw["mass_1"],
              0.0, 1.0, np.where(q1 >= 0, 1.0, -1.0))[:, None, :]

    x = np.zeros((m, N_OBJ_MAX, 7), dtype=np.float32)
    x[:, :4] = np.concatenate([had, lep], axis=1)
    counts = np.full(m, 4, dtype=np.int64)

    met = raw[res["met"]] if res["met"] else np.zeros(m)
    metphi = raw[res["metphi"]] if res["metphi"] else np.zeros(m)
    if not res["met"]:
        print("WARNING: no met branch -> conditions met=0 (neutrino info lost)")

    w = raw[res["genWeight"]].astype(np.float64) if res["genWeight"] else np.ones(m)
    if res["puweight"]:
        w = w * raw[res["puweight"]].astype(np.float64)
    print(f"{m} events, {(w < 0).mean():.1%} negative weights")
    return _finalize(x, counts, met, metphi, w, class_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", help="glob(s) of CROWN ntuple ROOT files")
    ap.add_argument("--output", required=True, help="output .npz path")
    ap.add_argument("--class-id", type=int, required=True, help="0=reference, 1=hypothesis")
    ap.add_argument("--tau-encoding", choices=["anonymous", "corner"], default="anonymous",
                    help="tau_h token type: anonymous=stock jet (0,0,0); corner=(0,0,+-1)")
    ap.add_argument("--smoke", type=int, default=0, help="fabricate N synthetic events (no ROOT)")
    ap.add_argument("--max-events", type=int, default=0)
    args = ap.parse_args()

    if args.smoke:
        out = make_smoke(args.smoke, args.class_id, tau_encoding=args.tau_encoding)
    else:
        if not args.input:
            ap.error("--input required unless --smoke is given")
        out = convert(args.input, args.class_id, args.tau_encoding, args.max_events)
    np.savez_compressed(args.output, **out)
    print(f"wrote {args.output}: " + ", ".join(f"{k}{v.shape}" for k, v in out.items()))


if __name__ == "__main__":
    main()
