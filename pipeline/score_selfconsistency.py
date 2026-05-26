#!/usr/bin/env python3
"""
score_selfconsistency.py — layer-2 self-consistency (Rev2 criterion 6).

Compares a Boltz-2 fold of the LigandMPNN-designed sequence to the ORIGINAL RFD2 backbone:
  * CA-RMSD (Kabsch superposition over matched CA atoms)  -- does the sequence fold back to
    the designed shape?
  * pLDDT (from Boltz confidence json)                    -- is the model confident?
Self-consistent (proxy) = CA-RMSD <= 2.0 A AND complex_plddt >= 0.70  (standard de novo bar).

Needs numpy (run inside the Boltz venv). Deterministic structured parsing only (PDB fixed
columns; mmCIF _atom_site loop) — no nuanced-content parsing.

Usage:
  # single (validate on the smoke):
  python score_selfconsistency.py --design <rfd2_design.pdb> --pred <fold.cif> --conf <confidence.json>
  # batch over a target dir laid out as <dir>/<design_id>/{design.pdb, boltz_out/.../*.cif + *.json}:
  python score_selfconsistency.py --batch <selfconsist/<target>>  [--out scores_sc.json]
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np

RMSD_PASS = 2.0
PLDDT_PASS = 0.70


def pdb_ca(path):
    out = []
    for line in open(path):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                out.append((int(line[22:26]), float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                continue
    out.sort(key=lambda r: r[0])
    return np.array([[x, y, z] for _, x, y, z in out], dtype=float)


def cif_ca(path):
    """Parse CA coords from an mmCIF _atom_site loop (column-order aware)."""
    cols, rows, in_loop, header = [], [], False, False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s == "loop_":
                cols, in_loop, header = [], True, False
                continue
            if in_loop and s.startswith("_atom_site."):
                cols.append(s.split(".", 1)[1]); header = True
                continue
            if header and not s.startswith("_atom_site."):
                header = False
                in_loop = bool(cols) and ("label_atom_id" in cols)
            if in_loop and s and not s.startswith("_") and s != "loop_":
                if s.startswith("#"):
                    in_loop = False
                    continue
                rows.append(s.split())
    if not cols or not rows:
        return np.empty((0, 3))
    idx = {c: i for i, c in enumerate(cols)}
    need = ("label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z")
    if not all(k in idx for k in need):
        return np.empty((0, 3))
    grp = idx.get("group_PDB")
    seq = idx.get("label_seq_id")
    recs = []
    for r in rows:
        if len(r) < len(cols):
            continue
        if r[idx["label_atom_id"]].strip('"') != "CA":
            continue
        if grp is not None and r[grp] not in ("ATOM", "."):
            continue
        try:
            key = int(r[seq]) if seq is not None and r[seq] not in (".", "?") else len(recs)
            recs.append((key, float(r[idx["Cartn_x"]]), float(r[idx["Cartn_y"]]), float(r[idx["Cartn_z"]])))
        except ValueError:
            continue
    recs.sort(key=lambda t: t[0])
    return np.array([[x, y, z] for _, x, y, z in recs], dtype=float)


def kabsch_rmsd(P, Q):
    """RMSD after optimal rigid superposition of P onto Q (equal-length, ordered)."""
    n = min(len(P), len(Q))
    if n == 0:
        return None, 0
    P, Q = P[:n], Q[:n]
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1, 1, d])
    R = V @ D @ Wt
    Pr = Pc @ R
    return float(np.sqrt(((Pr - Qc) ** 2).sum() / n)), n


def read_plddt(conf_path):
    if not conf_path or not os.path.isfile(conf_path):
        return {}
    d = json.load(open(conf_path))
    return {k: d.get(k) for k in ("confidence_score", "complex_plddt", "ptm", "iptm") if k in d}


def score_pair(design_pdb, pred_cif, conf_json):
    P = pdb_ca(design_pdb)            # RFD2 backbone
    Q = cif_ca(pred_cif)             # Boltz fold of the designed sequence
    rmsd, n = kabsch_rmsd(P, Q)
    conf = read_plddt(conf_json)
    plddt = conf.get("complex_plddt")
    res = {"design": os.path.basename(design_pdb), "ca_rmsd": round(rmsd, 3) if rmsd is not None else None,
           "n_ca_matched": n, "n_ca_design": len(P), "n_ca_pred": len(Q),
           "complex_plddt": plddt, "confidence_score": conf.get("confidence_score")}
    res["self_consistent"] = bool(rmsd is not None and rmsd <= RMSD_PASS
                                  and plddt is not None and plddt >= PLDDT_PASS)
    return res


def find_outputs(design_dir):
    """For a per-design dir: locate design.pdb, the Boltz .cif, and the confidence json."""
    design = os.path.join(design_dir, "design.pdb")
    cif = glob.glob(os.path.join(design_dir, "boltz_out", "**", "*_model_0.cif"), recursive=True)
    conf = glob.glob(os.path.join(design_dir, "boltz_out", "**", "confidence_*_model_0.json"), recursive=True)
    if os.path.isfile(design) and cif:
        return design, cif[0], (conf[0] if conf else None)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design"); ap.add_argument("--pred"); ap.add_argument("--conf")
    ap.add_argument("--batch"); ap.add_argument("--out")
    a = ap.parse_args()

    if a.batch:
        subdirs = [d for d in sorted(glob.glob(os.path.join(a.batch, "*"))) if os.path.isdir(d)]
        scores = []
        for d in subdirs:
            found = find_outputs(d)
            if found:
                scores.append(score_pair(*found))
        if not scores:
            raise SystemExit(f"no scorable per-design dirs under {a.batch}")
        ok = [s for s in scores if s["self_consistent"]]
        by_rmsd = sorted(scores, key=lambda s: (s["ca_rmsd"] is None, s["ca_rmsd"]))
        summary = {"target": os.path.basename(os.path.normpath(a.batch)), "n": len(scores),
                   "n_self_consistent": len(ok), "frac_self_consistent": round(len(ok)/len(scores), 3),
                   "best_ca_rmsd": by_rmsd[0]["ca_rmsd"],
                   "median_ca_rmsd": sorted(s["ca_rmsd"] for s in scores if s["ca_rmsd"] is not None)[len(scores)//2]
                       if any(s["ca_rmsd"] is not None for s in scores) else None,
                   "median_plddt": sorted(s["complex_plddt"] for s in scores if s["complex_plddt"] is not None)[len(scores)//2]
                       if any(s["complex_plddt"] is not None for s in scores) else None}
        out = a.out or os.path.join(a.batch, "scores_sc.json")
        json.dump({"summary": summary, "designs": scores}, open(out, "w"), indent=2)
        print(f"=== layer-2 self-consistency: {summary['target']} ({summary['n']} designs) ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"  wrote {out}")
        return

    if not (a.design and a.pred):
        raise SystemExit("need --design and --pred (and --conf), or --batch")
    res = score_pair(a.design, a.pred, a.conf)
    print("=== layer-2 self-consistency (single) ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print(f"\n  PASS if ca_rmsd<= {RMSD_PASS} and complex_plddt>= {PLDDT_PASS}  ->  "
          f"{'SELF-CONSISTENT' if res['self_consistent'] else 'not self-consistent'}")


if __name__ == "__main__":
    main()
