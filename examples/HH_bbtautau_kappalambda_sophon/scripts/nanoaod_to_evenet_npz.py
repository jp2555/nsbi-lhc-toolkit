"""CMS NanoAOD (HH->bbtautau, 2024/v15) -> EveNet NPZ (same contract as delphes_to_evenet_npz).

Emits the identical NPZ schema (x, conditions, num_sequential_vectors, event_weight,
classification) so the downstream chain (inject_systematic, EveNet preprocess, sweep,
eval_closure) is shared between Delphes and CMS full-sim inputs. Object mapping:

  * jets   -> hadronic group, btag = (disc >= --btag-wp), isLepton=0, charge=0;
              cleaned against selected taus and leptons (dR < 0.4)
  * tau_h  -> hadronic group (merged + pt-sorted with jets), btag=0, isLepton=0;
              charge: --tau-encoding anonymous -> 0 (stock), corner -> +-1 from Tau_charge
  * e / mu -> lepton group, isLepton=1, charge=+-1
  * MET    -> conditions[met, met_phi] from --met-collection (PuppiMET default)
  * weight -> genWeight (sign preserved)

Default object selection (flags override):
  tau:  pt>20, |eta|<2.3, DeepTau2018v2p5 VSjet>=5 (Medium), VSe>=2 (VVLoose), VSmu>=1 (VLoose)
  jet:  pt>20, |eta|<2.5
  ele:  pt>10, |eta|<2.5, cutBased>=2 (Loose) when the branch exists
  muon: pt>10, |eta|<2.4, looseId when the branch exists
Event preselection: >= --min-taus (2) taus and >= --min-jets (2) cleaned jets (bbtautau
tauh-tauh baseline; relax for lep-tau channels).

--btag-wp is REQUIRED in real mode: look up the era working point from the BTV tables for
--btag-branch (default Jet_btagUParTAK4B; auto-falls back to Jet_btagDeepFlavB with a warning
if the UParT branch is absent). No WP value is baked in on purpose.

VERIFY-ON-PERLMUTTER: branch names below target 2024 NanoAOD v15. Check once against a real
file (`uproot.open(f)['Events'].keys(filter_name='Tau_id*')`) before bulk conversion.

Smoke mode (`--smoke N`) reuses the shared fabricated-event generator (tau-aware).
"""
import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delphes_to_evenet_npz import (  # noqa: E402
    _finalize, _pack_event, make_smoke, tau_corner_charge)

DEFAULTS = dict(
    tau_pt=20.0, tau_eta=2.3, tau_vsjet=5, tau_vse=2, tau_vsmu=1,
    jet_pt=20.0, jet_eta=2.5, lep_pt=10.0, drclean=0.4,
)


def _np(arr, name, i):
    import awkward as ak
    return ak.to_numpy(arr[name][i])


def _has(arr, *names):
    return all(n in arr.fields for n in names)


def _dr2(eta1, phi1, eta2, phi2):
    dphi = np.mod(phi1 - phi2 + np.pi, 2 * np.pi) - np.pi
    return (eta1 - eta2) ** 2 + dphi ** 2


def _sel_taus(arr, i, a):
    pt = _np(arr, "Tau_pt", i)
    if not len(pt):
        return [np.zeros(0)] * 5
    eta, phi, m, q = (_np(arr, f"Tau_{f}", i) for f in ("eta", "phi", "mass", "charge"))
    keep = (pt > a.tau_pt) & (np.abs(eta) < a.tau_eta)
    keep &= _np(arr, a.tau_id_branch, i) >= a.tau_vsjet
    if _has(arr, a.tau_id_branch.replace("VSjet", "VSe")):
        keep &= _np(arr, a.tau_id_branch.replace("VSjet", "VSe"), i) >= a.tau_vse
    if _has(arr, a.tau_id_branch.replace("VSjet", "VSmu")):
        keep &= _np(arr, a.tau_id_branch.replace("VSjet", "VSmu"), i) >= a.tau_vsmu
    return pt[keep], eta[keep], phi[keep], m[keep], q[keep]


def _sel_leptons(arr, i, a):
    """-> list of (E, pt, eta, phi, 0, 1, charge) tuples, both flavours."""
    out = []
    for coll, etamax, idcut in (("Electron", 2.5, ("Electron_cutBased", 2)),
                                ("Muon", 2.4, ("Muon_looseId", 1))):
        if not _has(arr, f"{coll}_pt"):
            continue
        pt = _np(arr, f"{coll}_pt", i)
        if not len(pt):
            continue
        eta, phi, q = (_np(arr, f"{coll}_{f}", i) for f in ("eta", "phi", "charge"))
        keep = (pt > a.lep_pt) & (np.abs(eta) < etamax)
        if _has(arr, idcut[0]):
            keep &= _np(arr, idcut[0], i) >= idcut[1]
        for k in np.where(keep)[0]:
            out.append((pt[k] * np.cosh(eta[k]), pt[k], eta[k], phi[k], 0.0, 1.0, float(q[k])))
    return out


def _event_objects(arr, i, a, btag_branch):
    """-> (objs, n_taus, n_jets): hadronic group (jets+taus merged, pt-sorted) then leptons."""
    tpt, teta, tphi, tm, tq = _sel_taus(arr, i, a)
    leps = _sel_leptons(arr, i, a)

    jpt = _np(arr, "Jet_pt", i)
    jeta, jphi, jm = (_np(arr, f"Jet_{f}", i) for f in ("eta", "phi", "mass"))
    jb = _np(arr, btag_branch, i)
    keep = (jpt > a.jet_pt) & (np.abs(jeta) < a.jet_eta)
    for peta, pphi in ([(teta[k], tphi[k]) for k in range(len(tpt))]
                       + [(l[2], l[3]) for l in leps]):
        keep &= _dr2(jeta, jphi, peta, pphi) > a.drclean ** 2

    had = []
    for k in np.where(keep)[0]:
        E = np.sqrt((jpt[k] * np.cosh(jeta[k])) ** 2 + max(jm[k], 0.0) ** 2)
        had.append((E, jpt[k], jeta[k], jphi[k],
                    1.0 if jb[k] >= a.btag_wp else 0.0, 0.0, 0.0))
    n_jets = len(had)
    for k in range(len(tpt)):
        E = np.sqrt((tpt[k] * np.cosh(teta[k])) ** 2 + max(tm[k], 0.0) ** 2)
        q = tau_corner_charge(float(tq[k])) if a.tau_encoding == "corner" else 0.0
        had.append((E, tpt[k], teta[k], tphi[k], 0.0, 0.0, q))
    had.sort(key=lambda o: -o[1])
    leps.sort(key=lambda o: -o[1])
    return had + leps, len(tpt), n_jets


def _branch_list(a):
    """Explicit read list (NanoAOD has ~2000 branches; reading all is very slow).
    Unmatched names are silently skipped by uproot, which lets the fallbacks work."""
    tau_id = a.tau_id_branch
    return (
        [f"Tau_{f}" for f in ("pt", "eta", "phi", "mass", "charge")]
        + [tau_id, tau_id.replace("VSjet", "VSe"), tau_id.replace("VSjet", "VSmu")]
        + [f"Jet_{f}" for f in ("pt", "eta", "phi", "mass")]
        + [a.btag_branch, "Jet_btagDeepFlavB"]
        + [f"Electron_{f}" for f in ("pt", "eta", "phi", "charge", "cutBased")]
        + [f"Muon_{f}" for f in ("pt", "eta", "phi", "charge", "looseId")]
        + [f"{a.met_collection}_pt", f"{a.met_collection}_phi", "MET_pt", "MET_phi",
           "genWeight"]
    )


def convert_nanoaod(path, a, step_size=20000):
    import uproot
    xs, counts, mets, mphis, weights = [], [], [], [], []
    n_seen = n_kept = 0
    btag_branch, met = a.btag_branch, a.met_collection
    for arr in uproot.iterate(f"{path}:Events", step_size=step_size, library="ak",
                              filter_name=_branch_list(a)):
        if not _has(arr, "Tau_pt", "Jet_pt"):
            raise KeyError(f"Tau_pt/Jet_pt missing in {path}; not a NanoAOD Events tree?")
        if not _has(arr, a.tau_id_branch):
            raise KeyError(f"{a.tau_id_branch} missing; set --tau-id-branch to the file's "
                           "DeepTau VSjet branch (check Tau_id* keys).")
        if not _has(arr, btag_branch):
            fallback = "Jet_btagDeepFlavB"
            if not _has(arr, fallback):
                raise KeyError(f"neither {btag_branch} nor {fallback} present.")
            warnings.warn(f"{btag_branch} absent -> falling back to {fallback}; "
                          "make sure --btag-wp matches THIS discriminant.")
            btag_branch = fallback
        if not _has(arr, f"{met}_pt"):
            fallback = "MET"
            if not _has(arr, f"{fallback}_pt"):
                raise KeyError(f"neither {met}_pt nor {fallback}_pt present.")
            warnings.warn(f"{met}_pt absent -> falling back to {fallback}_pt/phi.")
            met = fallback

        n = len(arr["Jet_pt"])
        n_seen += n
        for i in range(n):
            objs, n_taus, n_jets = _event_objects(arr, i, a, btag_branch)
            if n_taus < a.min_taus or n_jets < a.min_jets:
                continue
            row, c = _pack_event(objs)
            xs.append(row); counts.append(c)
            mets.append(float(arr[f"{met}_pt"][i])); mphis.append(float(arr[f"{met}_phi"][i]))
            weights.append(float(arr["genWeight"][i]) if _has(arr, "genWeight") else 1.0)
        n_kept = len(xs)
    if not n_kept:
        raise RuntimeError(f"no events passed the preselection ({n_seen} read); "
                           "check WPs / --min-taus / --min-jets.")
    print(f"selected {n_kept}/{n_seen} events "
          f"({a.min_taus}+ taus [VSjet>={a.tau_vsjet}], {a.min_jets}+ cleaned jets)")
    return _finalize(np.stack(xs), counts, mets, mphis, weights, a.class_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="NanoAOD ROOT path/glob (real mode)")
    ap.add_argument("--output", required=True, help="output .npz path")
    ap.add_argument("--class-id", type=int, required=True, help="0=reference, 1=hypothesis")
    ap.add_argument("--smoke", type=int, default=0, help="fabricate N synthetic events (no ROOT)")
    ap.add_argument("--step-size", type=int, default=20000)
    ap.add_argument("--tau-encoding", choices=["anonymous", "corner"], default="anonymous",
                    help="tau_h token type: anonymous=stock jet (0,0,0); corner=(0,0,+-1)")
    ap.add_argument("--btag-branch", default="Jet_btagUParTAK4B")
    ap.add_argument("--btag-wp", type=float, default=None,
                    help="era WP for --btag-branch (BTV tables); REQUIRED in real mode")
    ap.add_argument("--tau-id-branch", default="Tau_idDeepTau2018v2p5VSjet")
    ap.add_argument("--met-collection", default="PuppiMET")
    ap.add_argument("--min-taus", type=int, default=2)
    ap.add_argument("--min-jets", type=int, default=2)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = ap.parse_args()

    if args.smoke:
        out = make_smoke(args.smoke, args.class_id, tau_encoding=args.tau_encoding)
    else:
        if not args.input:
            ap.error("--input required unless --smoke is given")
        if args.btag_wp is None:
            ap.error("--btag-wp is required in real mode (era WP from the BTV tables)")
        out = convert_nanoaod(args.input, args, args.step_size)
    np.savez_compressed(args.output, **out)
    print(f"wrote {args.output}: " + ", ".join(f"{k}{v.shape}" for k, v in out.items()))


if __name__ == "__main__":
    main()
