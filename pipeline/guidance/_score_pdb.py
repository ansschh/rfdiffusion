#!/usr/bin/env python3
"""_score_pdb.py - score a PDB file against an A_cat using E_path (and later
other E_cat terms). Used for term-by-term sanity checks before locking.

Usage:
  python _score_pdb.py <a_cat.json> <protein.pdb> [--apex-back 0.0|1.6]
"""
from __future__ import annotations
import argparse, os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))  # pipeline/

from guidance.a_cat_fields import load as load_fields
from guidance.e_terms.path import e_path


def parse_pdb(path):
    atoms = []
    for line in open(path):
        if line[:6].strip() not in ("ATOM", "HETATM"):
            continue
        try:
            name = line[12:16].strip()
            el = line[76:78].strip().upper() or "".join(c for c in name if c.isalpha())[:2].upper()
            atoms.append({
                "record": line[:6].strip(), "name": name, "element": el,
                "resname": line[17:20].strip(), "chain": line[21].strip(),
                "resseq": int(line[22:26]),
                "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
            })
        except ValueError:
            continue
    return atoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a_cat_json")
    ap.add_argument("pdb", nargs="+", help="One or more PDB files to compare")
    ap.add_argument("--apex-back", type=float, default=0.0,
                    help="path_apex_offset_back_A (0.0 = apex at hydride; 1.6 ~ V_rxn cone for 3ZP9)")
    ap.add_argument("--sigma-perp", type=float, default=0.5)
    ap.add_argument("--sigma-par", type=float, default=0.5)
    ap.add_argument("--per-atom", action="store_true")
    args = ap.parse_args()

    fields = load_fields(args.a_cat_json,
                         path_apex_offset_back=args.apex_back,
                         path_sigma_perp=args.sigma_perp,
                         path_sigma_par=args.sigma_par)
    print(f"# A_cat = {args.a_cat_json}  target={fields.target}  mode={fields.mode}")
    print(f"# E_path tunables: apex_back={args.apex_back}  sigma_perp={args.sigma_perp}  sigma_par={args.sigma_par}")
    print(f"# {'PDB':>50}  {'E_path':>10}  {'n_atoms':>8}")
    print(f"# {'-'*50}  {'-'*10}  {'-'*8}")
    for p in args.pdb:
        atoms = parse_pdb(p)
        if args.per_atom:
            E, per = e_path(atoms, fields, return_per_atom=True)
            print(f"  {os.path.basename(p):>50}  {E:10.4f}  {len(atoms):8d}")
            for a in sorted(per, key=lambda x: -x["e_i"])[:10]:
                print(f"     -> {a['resname']}{a['resseq']}{a['chain']}/{a['name']:<4}  occ={a['occ']}  e={a['e_i']}")
        else:
            E = e_path(atoms, fields)
            print(f"  {os.path.basename(p):>50}  {E:10.4f}  {len(atoms):8d}")


if __name__ == "__main__":
    main()
