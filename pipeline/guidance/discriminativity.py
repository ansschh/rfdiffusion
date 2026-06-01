#!/usr/bin/env python3
"""discriminativity.py - score one protein under real and damaged A_cat
variants. Outputs a per-term and total-E_cat matrix. Used to validate that
E_cat assigns LOWER energy to (real protein, real A_cat) than to (real
protein, damaged A_cat).

Usage:
  python discriminativity.py pipeline/compiled/3ZP9/A_cat.json pipeline/pdb/3ZP9.pdb
  python discriminativity.py pipeline/compiled/3ZP9/A_cat.json pipeline/pdb/3ZP9.pdb --out report.json
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guidance.a_cat_fields import ACatFields
from guidance.e_cat import e_cat, DEFAULT_LAMBDAS
from guidance.damaged_controls import generate_all
from guidance._score_pdb import parse_pdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a_cat_json")
    ap.add_argument("pdb")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lambdas", default=None, help="JSON dict overriding lambda weights")
    args = ap.parse_args()

    base = json.load(open(args.a_cat_json))
    variants = generate_all(base)
    atoms = parse_pdb(args.pdb)
    lambdas = json.loads(args.lambdas) if args.lambdas else DEFAULT_LAMBDAS

    rows = []
    print(f"# A_cat   = {args.a_cat_json}  target={base.get('target')}  mode={base.get('mode')}")
    print(f"# protein = {args.pdb}  n_atoms={len(atoms)}")
    print(f"# lambdas = {lambdas}")
    print()
    cols = ("path", "contact", "avoid", "anchor", "seq_chem", "site",
            "face", "coord_zones", "elec_field", "dynamics")
    hdr = "{:<22}".format("variant") + "".join("{:>9}".format("E_" + c) for c in cols) + "  {:>10}".format("E_cat")
    print(hdr)
    print("-" * len(hdr))
    for name, a_var in variants.items():
        fields = ACatFields(a_var)
        E, br = e_cat(atoms, fields, lambdas=lambdas, return_breakdown=True)
        t = br["terms"]
        row = {"variant": name, "E_cat": round(E, 4),
               **{f"E_{k}": round(t.get(k, 0.0), 4) for k in cols}}
        rows.append(row)
        line = "{:<22}".format(name) + "".join("{:>9.3f}".format(t.get(c, 0.0)) for c in cols) + "  {:>10.3f}".format(E)
        print(line)

    # discrimination summary
    real = next(r for r in rows if r["variant"] == "real")
    print()
    print(f"# Real E_cat = {real['E_cat']:.3f}")
    print(f"# Damaged E_cat range: [{min(r['E_cat'] for r in rows if r['variant']!='real'):.3f}, {max(r['E_cat'] for r in rows if r['variant']!='real'):.3f}]")
    n_worse = sum(1 for r in rows if r["variant"] != "real" and r["E_cat"] > real["E_cat"])
    print(f"# Damaged variants with E_cat > real: {n_worse}/{len(rows)-1}")

    if args.out:
        json.dump({"target": base.get("target"), "mode": base.get("mode"),
                   "protein": args.pdb, "lambdas": lambdas, "rows": rows},
                  open(args.out, "w"), indent=2)
        print(f"# wrote {args.out}")


if __name__ == "__main__":
    main()
