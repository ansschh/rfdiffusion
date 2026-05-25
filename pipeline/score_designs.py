#!/usr/bin/env python3
"""
score_designs.py — deterministic motif-preservation scoring (Rev2 layer 1) for RFD2 ArM designs.

Runs on the RFD2 output PDBs + the compiled manifest. NO folding — these are the structural
checks computable directly on the generated backbone+cofactor. The deeper self-consistency
layer (LigandMPNN -> Boltz-2 co-fold -> does the metal site re-form) is layer 2, separate.

Per design (all by deterministic geometry):
  * metal_present         : the catalytic metal (by element) survived into the design
  * buried_core_frac      : fraction of cofactor heavy atoms with >=1 protein heavy atom <5A
                            (is the cofactor enclosed in a pocket? rasa=0 should bury it)
  * pocket_residues       : # distinct protein residues lining the cofactor (any atom <5A)
  * burial_count          : # protein heavy atoms within 6A of the metal
  * min_contact           : closest protein-heavy <-> cofactor-heavy distance
  * clashes               : # non-bonded protein/cofactor heavy-atom pairs < 2.0A (fatal overlap)
  * backbone_breaks       : # consecutive CA-CA distances > 4.5A (chain discontinuities)
  * n_residues            : CA count
Layer-1 verdict 'preserved' = metal present AND buried_core_frac>=0.6 AND clashes==0
AND backbone_breaks<=2. A PROXY for Rev2 criteria 1/2/5; criterion 6 (self-consistency)
needs layer 2.

NB (Rev3): the wrong_metal control is geometrically identical to the real target (Ir->Zn,
same coords) so it will NOT separate at layer 1 by design — it is meant to separate at
layer 2 (Boltz chemistry). scramble_guideposts CAN separate at layer 1 (worse burial/pocket).

Usage:
  python score_designs.py <design_dir> [--manifest <manifest.json>] [--out scores.json]
  python score_designs.py --compare scores_real.json scores_wrong.json scores_scramble.json
"""
from __future__ import annotations
import argparse, glob, json, math, os

METALS = {"IR", "ZN", "RH", "RU", "FE", "MN", "CU", "CO", "NI", "PD", "PT", "MO", "W", "OS", "V", "CR"}


def parse_pdb(path):
    prot, het = [], []   # (resseq, name, element, x, y, z, resname, record)
    with open(path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            try:
                name = line[12:16].strip()
                el = line[76:78].strip().upper() or "".join(c for c in name if c.isalpha())[:2].upper()
                resname = line[17:20].strip()
                resseq = int(line[22:26])
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            rec_t = (resseq, name, el, x, y, z, resname, rec)
            (prot if rec == "ATOM" else het).append(rec_t)
    return prot, het


def d(a, b):
    return math.sqrt((a[3]-b[3])**2 + (a[4]-b[4])**2 + (a[5]-b[5])**2)


def score_one(path):
    prot, het = parse_pdb(path)
    cofactor = [a for a in het if a[6] != "ORI"]              # LIG core (+ any non-ORI het)
    cof_heavy = [a for a in cofactor if a[2] != "H"]
    prot_heavy = [a for a in prot if a[2] != "H"]
    metals = [a for a in cofactor if a[2] in METALS]
    res = {"file": os.path.basename(path), "n_residues": len({a[0] for a in prot if a[1] == "CA"}),
           "n_cofactor_heavy": len(cof_heavy), "metal_present": bool(metals),
           "metal_element": metals[0][2] if metals else None}
    if not cof_heavy or not prot_heavy:
        res.update({"buried_core_frac": 0.0, "pocket_residues": 0, "burial_count": 0,
                    "min_contact": None, "clashes": 0, "backbone_breaks": 0, "preserved": False})
        return res

    # burial: cofactor heavy atoms with >=1 protein heavy atom within 5A
    buried = sum(1 for c in cof_heavy if any(d(c, p) < 5.0 for p in prot_heavy))
    res["buried_core_frac"] = round(buried / len(cof_heavy), 3)
    # pocket lining residues
    lining = {p[0] for p in prot_heavy if any(d(p, c) < 5.0 for c in cof_heavy)}
    res["pocket_residues"] = len(lining)
    # protein heavy atoms within 6A of the metal
    if metals:
        m = metals[0]
        res["burial_count"] = sum(1 for p in prot_heavy if d(p, m) < 6.0)
    else:
        res["burial_count"] = 0
    # closest contact + clashes (<2.0A non-bonded heavy-heavy)
    mind, clash = 1e9, 0
    for c in cof_heavy:
        for p in prot_heavy:
            dd = d(c, p)
            mind = min(mind, dd)
            if dd < 2.0:
                clash += 1
    res["min_contact"] = round(mind, 2)
    res["clashes"] = clash
    # backbone continuity
    cas = sorted([a for a in prot if a[1] == "CA"], key=lambda a: a[0])
    breaks = sum(1 for i in range(1, len(cas)) if d(cas[i-1], cas[i]) > 4.5)
    res["backbone_breaks"] = breaks
    res["preserved"] = bool(metals) and res["buried_core_frac"] >= 0.6 \
        and clash == 0 and breaks <= 2
    return res


def summarize(target, scores):
    n = len(scores)
    preserved = [s for s in scores if s["preserved"]]
    by_burial = sorted(scores, key=lambda s: s["buried_core_frac"], reverse=True)
    top5 = by_burial[:5]
    def med(key):
        vals = sorted(s[key] for s in scores if s[key] is not None)
        return round(vals[len(vals)//2], 3) if vals else None
    return {
        "target": target, "n_designs": n,
        "n_preserved": len(preserved),
        "frac_preserved": round(len(preserved)/n, 3) if n else 0,
        "best_of_N_buried_core_frac": by_burial[0]["buried_core_frac"] if n else None,
        "top5_mean_buried_core_frac": round(sum(s["buried_core_frac"] for s in top5)/len(top5), 3) if top5 else None,
        "median_buried_core_frac": med("buried_core_frac"),
        "median_pocket_residues": med("pocket_residues"),
        "median_clashes": med("clashes"),
        "any_clash_free": sum(1 for s in scores if s["clashes"] == 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design_dir", nargs="?")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs="+", help="summary JSONs to tabulate real-vs-control")
    a = ap.parse_args()

    if a.compare:
        rows = [json.load(open(p))["summary"] for p in a.compare]
        cols = ["target", "n_designs", "frac_preserved", "best_of_N_buried_core_frac",
                "top5_mean_buried_core_frac", "median_buried_core_frac",
                "median_pocket_residues", "median_clashes"]
        print(" | ".join(f"{c:>26}" if c == "target" else f"{c:>22}" for c in cols))
        for r in rows:
            print(" | ".join(f"{str(r.get(c)):>26}" if c == "target" else f"{str(r.get(c)):>22}" for c in cols))
        print("\nRev3 read: real should beat scramble_guideposts on burial/pocket. wrong_metal is\n"
              "geometrically identical here -> expect it to match real at layer 1; it separates at\n"
              "layer 2 (Boltz). If scramble matches real, layer-1 burial is not discriminative.")
        return

    pdbs = sorted(glob.glob(os.path.join(a.design_dir, "*-atomized-bb-False.pdb")))
    pdbs = [p for p in pdbs if "/unidealized/" not in p.replace("\\", "/")]
    if not pdbs:
        raise SystemExit(f"no design PDBs (*-atomized-bb-False.pdb) in {a.design_dir}")
    target = os.path.basename(os.path.normpath(a.design_dir))
    scores = [score_one(p) for p in pdbs]
    summary = summarize(target, scores)
    out = a.out or os.path.join(a.design_dir, "scores.json")
    json.dump({"summary": summary, "designs": scores}, open(out, "w"), indent=2)

    print(f"=== layer-1 motif-preservation: {target}  ({len(scores)} designs) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n  metal retained in {sum(1 for s in scores if s['metal_present'])}/{len(scores)} designs"
          f" (element: {next((s['metal_element'] for s in scores if s['metal_element']), '?')})")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
