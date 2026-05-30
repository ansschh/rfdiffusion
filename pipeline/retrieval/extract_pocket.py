#!/usr/bin/env python3
"""extract_pocket.py — extract a typed protein pocket around a metal cofactor.

For retrieval: each library entry is a typed pocket in absolute (world) coordinates,
plus the metal position. The retrieval scorer aligns A_cat's local frame onto each
candidate pocket and computes typed-Gaussian overlap.

Usage:
  python extract_pocket.py <pdb_file> --metal IR
  python extract_pocket.py <pdb_file> --metal ZN --metal-resname ZN --out pocket.json
"""
from __future__ import annotations
import argparse, json, math, os
from collections import defaultdict

METALS = {"IR","ZN","RH","RU","FE","MN","CU","CO","NI","PD","PT","MO","W","OS","V","CR","MG","CA","NA","K","AL"}
BACKBONE = {"N","CA","C","O","OXT","H"}
RESIDUE_TYPE = {
    "ALA":"hydrophobic","VAL":"hydrophobic","LEU":"hydrophobic","ILE":"hydrophobic",
    "MET":"hydrophobic","PRO":"hydrophobic","GLY":"small",
    "PHE":"aromatic","TYR":"aromatic","TRP":"aromatic",
    "SER":"polar","THR":"polar","ASN":"polar","GLN":"polar","CYS":"polar",
    "HIS":"anchor","LYS":"charged_base","ARG":"charged_base",
    "ASP":"charged_acid","GLU":"charged_acid",
}


def parse_pdb(path):
    out = []
    for line in open(path):
        if line[:6].strip() not in ("ATOM","HETATM"): continue
        try:
            name = line[12:16].strip()
            el = line[76:78].strip().upper() or "".join(c for c in name if c.isalpha())[:2].upper()
            out.append({"record": line[:6].strip(), "name": name, "element": el,
                        "resname": line[17:20].strip(), "chain": line[21].strip(),
                        "resseq": int(line[22:26]),
                        "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
                        "occ": float(line[54:60] or "1.0")})
        except ValueError: continue
    return out


def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def extract_pocket(pdb_path, metal_element, metal_resname=None, max_dist=8.0):
    atoms = parse_pdb(pdb_path)
    metals = [a for a in atoms if a["element"]==metal_element.upper() and
              (metal_resname is None or a["resname"]==metal_resname.upper())]
    if not metals:
        raise SystemExit(f"metal {metal_element} ({metal_resname or 'any resname'}) not found in {pdb_path}")
    metal = max(metals, key=lambda a: a.get("occ", 1.0))
    metal_xyz = (metal["x"], metal["y"], metal["z"])

    by_res = defaultdict(list)
    for a in atoms:
        if a["record"] == "ATOM" and a["resname"] in RESIDUE_TYPE:
            by_res[(a["chain"], a["resseq"])].append(a)

    pocket = []
    for (chain, resseq), res_atoms in by_res.items():
        resname = res_atoms[0]["resname"]
        sc = [a for a in res_atoms if a["element"] != "H" and a["name"] not in BACKBONE]
        if not sc:
            sc = [a for a in res_atoms if a["element"] != "H"]
        if not sc:
            continue
        min_d = min(dist((a["x"],a["y"],a["z"]), metal_xyz) for a in sc)
        if min_d > max_dist:
            continue
        n = len(sc)
        cx = sum(a["x"] for a in sc)/n
        cy = sum(a["y"] for a in sc)/n
        cz = sum(a["z"] for a in sc)/n
        pocket.append({
            "chain": chain, "resseq": resseq, "resname": resname,
            "type": RESIDUE_TYPE.get(resname, "other"),
            "sidechain_centroid_world": [round(cx, 3), round(cy, 3), round(cz, 3)],
            "min_dist_to_metal": round(min_d, 3),
            "n_sidechain_heavies": n,
        })
    pocket.sort(key=lambda p: p["min_dist_to_metal"])
    return {
        "pdb_path": pdb_path,
        "pdb_id": os.path.splitext(os.path.basename(pdb_path))[0].upper(),
        "metal": {"element": metal["element"], "resname": metal["resname"],
                  "world": [round(c, 3) for c in metal_xyz]},
        "pocket_residues": pocket,
        "n_residues": len(pocket),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb_path")
    ap.add_argument("--metal", required=True, help="Metal element symbol (e.g. IR, ZN, FE)")
    ap.add_argument("--metal-resname", help="Optional restrict by HETATM resname")
    ap.add_argument("--max-dist", type=float, default=8.0)
    ap.add_argument("--out")
    args = ap.parse_args()
    p = extract_pocket(args.pdb_path, args.metal, args.metal_resname, args.max_dist)
    out = args.out or args.pdb_path.replace(".pdb", "_pocket.json")
    json.dump(p, open(out, "w"), indent=2)
    print(f"=== pocket extracted: {p['pdb_id']} (metal={p['metal']['element']}/{p['metal']['resname']}) ===")
    print(f"  n_pocket_residues = {p['n_residues']}")
    type_counts = defaultdict(int)
    for r in p["pocket_residues"]:
        type_counts[r["type"]] += 1
    for t, n in sorted(type_counts.items()):
        print(f"    {t}: {n}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
