"""Delphes full-output ROOT -> padded per-jet constituent clouds + object tokens.

Reads the Delphes tree (Jet/EFlow*/MissingET/Electron/Muon/Event.Weight), assigns
EFlow constituents to AK4 jets, builds the per-jet padded cloud and per-event object
tokens (taus/leptons/MET), and writes a compressed .npz with arrays matching
WeightedParticleCloudDataset. Feature columns follow cloud_spec (F_PART=8, F_OBJ=6);
the sophon-ak4 integration (Task 11) extends F_PART to the checkpoint's full schema.

This baseline maps EFlow.JetIndex -> jet; for real Delphes without an explicit
constituent->jet index, replace assign_constituents() with a Delta-R association.
"""
import argparse
import numpy as np
import awkward as ak
import uproot

N_JETS_MAX, N_PART_MAX, F_PART, N_OBJ_MAX, F_OBJ = 4, 64, 8, 6, 6


def _logpt(pt):
    return np.log(np.clip(pt, 1e-3, None)).astype(np.float32)


def convert_tree(path, tree_name, out_path):
    arr = uproot.open(f"{path}:{tree_name}").arrays(library="ak")
    n = len(arr["Jet.PT"])
    parts = np.zeros((n, N_JETS_MAX, N_PART_MAX, F_PART), dtype=np.float32)
    part_mask = np.zeros((n, N_JETS_MAX, N_PART_MAX), dtype=np.float32)
    jet_mask = np.zeros((n, N_JETS_MAX), dtype=np.float32)
    obj = np.zeros((n, N_OBJ_MAX, F_OBJ), dtype=np.float32)
    obj_mask = np.zeros((n, N_OBJ_MAX), dtype=np.float32)

    for i in range(n):
        njet = min(len(arr["Jet.PT"][i]), N_JETS_MAX)
        jet_eta = ak.to_numpy(arr["Jet.Eta"][i]); jet_phi = ak.to_numpy(arr["Jet.Phi"][i])
        for j in range(njet):
            jet_mask[i, j] = 1.0
        # assign constituents to jets via EFlow.JetIndex
        cp = ak.to_numpy(arr["EFlow.PT"][i]); ce = ak.to_numpy(arr["EFlow.Eta"][i])
        cph = ak.to_numpy(arr["EFlow.Phi"][i]); cj = ak.to_numpy(arr["EFlow.JetIndex"][i])
        cc = ak.to_numpy(arr["EFlow.Charge"][i])
        counts = np.zeros(N_JETS_MAX, dtype=int)
        for k in range(len(cp)):
            j = int(cj[k])
            if j < 0 or j >= njet or counts[j] >= N_PART_MAX:
                continue
            p = counts[j]; counts[j] += 1
            part_mask[i, j, p] = 1.0
            deta = ce[k] - jet_eta[j]
            dphi = np.arctan2(np.sin(cph[k] - jet_phi[j]), np.cos(cph[k] - jet_phi[j]))
            parts[i, j, p] = [_logpt(cp[k]), deta, dphi, cc[k], 0.0, 0.0, 0.0,
                              1.0 if cc[k] != 0 else 0.0]
        # MET object token (type_id=3)
        obj_mask[i, 0] = 1.0
        met = float(arr["MissingET.MET"][i][0]) if len(arr["MissingET.MET"][i]) else 0.0
        mphi = float(arr["MissingET.Phi"][i][0]) if len(arr["MissingET.Phi"][i]) else 0.0
        obj[i, 0] = [_logpt(met), 0.0, np.sin(mphi), np.cos(mphi), met, 3.0]

    w = ak.to_numpy(arr["Event.Weight"]).astype(np.float64)
    np.savez_compressed(out_path, parts=parts, part_mask=part_mask, jet_mask=jet_mask,
                        obj=obj, obj_mask=obj_mask, w=w)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tree", default="Delphes")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    convert_tree(args.input, args.tree, args.output)
