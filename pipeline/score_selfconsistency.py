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
METAL_POSE_PASS = 2.0   # Rev2 crit-1: metal placed within 2.0 A of the design site (after CA fit)
METALS = {"IR", "ZN", "RH", "RU", "FE", "MN", "CU", "CO", "NI", "PD", "PT", "MO", "W", "OS", "V", "CR"}


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
    """CA coords from an mmCIF. Prefer biopython (robust); fall back to the hand parser."""
    try:
        from Bio.PDB import MMCIFParser
        s = MMCIFParser(QUIET=True).get_structure("x", path)
        out = []
        for model in s:
            for chain in model:
                for res in chain:
                    if res.has_id("CA"):
                        c = res["CA"].coord
                        out.append((res.id[1], float(c[0]), float(c[1]), float(c[2])))
            break  # first model only
        out.sort(key=lambda r: r[0])
        if out:
            return np.array([[x, y, z] for _, x, y, z in out], dtype=float)
    except Exception:
        pass
    return _cif_ca_manual(path)


def _cif_ca_manual(path):
    """Fallback: parse CA coords from an mmCIF _atom_site loop (column-order aware)."""
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


def kabsch_fit(P, Q):
    """Like kabsch_rmsd but also returns the rigid map. R, Pm, Qm satisfy (P-Pm)@R ~= (Q-Qm),
    so a PRED-frame point q maps into the DESIGN frame via (q-Qm)@R.T + Pm."""
    n = min(len(P), len(Q))
    if n == 0:
        return None, None, None, None, 0
    P, Q = P[:n], Q[:n]
    Pm, Qm = P.mean(0), Q.mean(0)
    Pc, Qc = P - Pm, Q - Qm
    V, S, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1, 1, d]) @ Wt
    rmsd = float(np.sqrt(((Pc @ R - Qc) ** 2).sum() / n))
    return rmsd, R, Pm, Qm, n


def metal_from_pdb(path):
    """Coordinates of the (first) catalytic metal heavy atom in an RFD2 design PDB."""
    for line in open(path):
        if line[:6].strip() not in ("ATOM", "HETATM"):
            continue
        name = line[12:16].strip()
        el = line[76:78].strip().upper() or "".join(c for c in name if c.isalpha())[:2].upper()
        if el in METALS:
            try:
                return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                pass
    return None


def metal_from_cif(path):
    """Coordinates of the (first) metal atom in a Boltz mmCIF fold (biopython, manual fallback)."""
    try:
        from Bio.PDB import MMCIFParser
        s = MMCIFParser(QUIET=True).get_structure("x", path)
        for model in s:
            for chain in model:
                for res in chain:
                    for atom in res:
                        if (atom.element or "").upper() in METALS:
                            c = atom.coord
                            return np.array([float(c[0]), float(c[1]), float(c[2])])
            break
    except Exception:
        pass
    # manual: scan _atom_site for a type_symbol in METALS
    cols, in_loop = [], False
    try:
        for line in open(path):
            t = line.strip()
            if t == "loop_":
                cols, in_loop = [], True; continue
            if in_loop and t.startswith("_atom_site."):
                cols.append(t.split(".", 1)[1]); continue
            if cols and t and not t.startswith("_") and not t.startswith("#"):
                r = t.split()
                if len(r) < len(cols):
                    continue
                idx = {c: i for i, c in enumerate(cols)}
                if "type_symbol" not in idx:
                    return None
                if r[idx["type_symbol"]].strip('"').upper() in METALS:
                    return np.array([float(r[idx["Cartn_x"]]), float(r[idx["Cartn_y"]]), float(r[idx["Cartn_z"]])])
    except Exception:
        pass
    return None


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
    ap.add_argument("--batch"); ap.add_argument("--target-dir", dest="target_dir"); ap.add_argument("--out")
    a = ap.parse_args()

    if a.target_dir:
        # Layout from selfconsist.sbatch: <dir>/designs/<id>.pdb  +  <dir>/folds/**/predictions/<id>__s<i>/
        # Per design = best-of-N over its sequences. Design passes if ANY sequence folds back
        # with ca_rmsd<=2.0 AND that fold's plddt>=0.70 (standard de novo self-consistency).
        from collections import defaultdict
        base = a.target_dir
        groups = defaultdict(list)
        for pd in glob.glob(os.path.join(base, "folds", "**", "predictions", "*"), recursive=True):
            name = os.path.basename(pd)
            if "__s" not in name:
                continue
            did = name.rsplit("__s", 1)[0]   # rsplit: the target tag itself can contain "__s" (e.g. 3ZP9__scramble_guideposts)
            cif = glob.glob(os.path.join(pd, "*_model_0.cif"))
            conf = glob.glob(os.path.join(pd, "confidence_*_model_0.json"))
            if cif:
                groups[did].append((cif[0], conf[0] if conf else None))
        rows = []
        for did, folds in sorted(groups.items()):
            dpdb = os.path.join(base, "designs", did + ".pdb")
            if not os.path.isfile(dpdb):
                continue
            P = pdb_ca(dpdb)
            dmetal = metal_from_pdb(dpdb)        # design metal site (if the cofactor metal is present)
            seqres = []
            for cif, conf in folds:
                r, R, Pm, Qm, _ = kabsch_fit(P, cif_ca(cif))
                md = None                        # metal-site displacement after CA superposition
                if dmetal is not None and R is not None:
                    qm = metal_from_cif(cif)
                    if qm is not None:
                        md = float(np.linalg.norm((qm - Qm) @ R.T + Pm - dmetal))
                seqres.append({"ca_rmsd": round(r, 3) if r is not None else None,
                               "plddt": read_plddt(conf).get("complex_plddt"),
                               "metal_disp": round(md, 3) if md is not None else None})
            best_rmsd = min((s["ca_rmsd"] for s in seqres if s["ca_rmsd"] is not None), default=None)
            best_plddt = max((s["plddt"] for s in seqres if s["plddt"] is not None), default=None)
            best_md = min((s["metal_disp"] for s in seqres if s["metal_disp"] is not None), default=None)
            passed = any(s["ca_rmsd"] is not None and s["ca_rmsd"] <= RMSD_PASS
                         and s["plddt"] is not None and s["plddt"] >= PLDDT_PASS for s in seqres)
            rows.append({"design": did, "n_seqs": len(seqres), "best_ca_rmsd": best_rmsd,
                         "best_plddt": round(best_plddt, 3) if best_plddt is not None else None,
                         "best_metal_disp": best_md, "self_consistent": passed})
        if not rows:
            raise SystemExit(f"no scorable designs under {base} (need designs/*.pdb + folds/**/predictions/*__s*)")
        ok = [r for r in rows if r["self_consistent"]]
        rmsds = sorted(r["best_ca_rmsd"] for r in rows if r["best_ca_rmsd"] is not None)
        mds = sorted(r["best_metal_disp"] for r in rows if r["best_metal_disp"] is not None)
        summary = {"target": os.path.basename(os.path.normpath(base)), "n_designs": len(rows),
                   "n_self_consistent": len(ok), "frac_self_consistent": round(len(ok)/len(rows), 3),
                   "best_ca_rmsd_overall": rmsds[0] if rmsds else None,
                   "median_best_ca_rmsd": rmsds[len(rmsds)//2] if rmsds else None,
                   # metal-site (Rev2 crit-1): only populated when the metal was co-folded
                   "n_metal_found": len(mds),
                   "best_metal_disp_overall": mds[0] if mds else None,
                   "median_best_metal_disp": mds[len(mds)//2] if mds else None,
                   "n_metal_site_within_2A": sum(1 for r in rows if r["best_metal_disp"] is not None
                                                 and r["best_metal_disp"] <= METAL_POSE_PASS)}
        json.dump({"summary": summary, "designs": rows}, open(os.path.join(base, "scores_sc.json"), "w"), indent=2)
        print(f"=== layer-2 self-consistency (best-of-N): {summary['target']} ===")
        for r in rows:
            print(f"  {r['design']}: best_rmsd={r['best_ca_rmsd']} best_plddt={r['best_plddt']} "
                  f"best_metal_disp={r['best_metal_disp']} n_seqs={r['n_seqs']} "
                  f"-> {'PASS' if r['self_consistent'] else 'no'}")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return

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
