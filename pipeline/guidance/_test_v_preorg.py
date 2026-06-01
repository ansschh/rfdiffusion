#!/usr/bin/env python3
"""Test ANM V_preorg on natural 3ZP9 vs synthetic disrupted designs.

Natural 3ZP9 is a real CA-II ArM scaffold — should be well-ordered (low RMSF).
We compare to:
  - z-flipped 3ZP9 (residues on wrong handedness side; should still be ordered
    BUT active-site CAs identified will differ)
  - Truncated 3ZP9 (drop 20% of residues randomly — simulates poor design)
"""
from __future__ import annotations
import argparse, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guidance.a_cat_fields import ACatFields
from guidance.e_terms.v_preorg import e_v_preorg
from guidance._score_pdb import parse_pdb


def truncate_random_residues(atoms, frac=0.2, seed=42):
    """Remove a random fraction of residues. Simulates a sparse design."""
    rng = random.Random(seed)
    by_res = {}
    for a in atoms:
        if a.get("record") != "ATOM": continue
        key = (a["chain"], a["resseq"])
        by_res.setdefault(key, []).append(a)
    keys = list(by_res.keys())
    n_drop = int(len(keys) * frac)
    drop = set(rng.sample(keys, n_drop))
    out = [a for a in atoms if a.get("record") != "ATOM" or
           (a["chain"], a["resseq"]) not in drop]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a_cat_json")
    ap.add_argument("pdb")
    args = ap.parse_args()

    a = json.load(open(args.a_cat_json))
    fields = ACatFields(a)
    atoms = parse_pdb(args.pdb)
    print(f"# A_cat={args.a_cat_json}  protein={args.pdb}  n_atoms={len(atoms)}")
    print()

    scenarios = [
        ("real (full 3ZP9)",                  atoms),
        ("random 20% residues removed",       truncate_random_residues(atoms, 0.2, seed=42)),
        ("random 40% residues removed",       truncate_random_residues(atoms, 0.4, seed=42)),
    ]
    fmt = "{:<40} {:>10} {:>12} {:>14} {:>10}"
    print(fmt.format("scenario", "n_CA", "n_active", "active_RMSF", "E_penalty"))
    print(fmt.format("-"*40, "-"*10, "-"*12, "-"*14, "-"*10))
    for name, ats in scenarios:
        E, br = e_v_preorg(ats, fields, return_breakdown=True)
        n_ca = br.get("n_CA", "-")
        n_act = br.get("n_active_site_CA", "-")
        rmsf = br.get("active_site_RMSF", "-")
        e_pen = br.get("E_penalty", round(E, 4))
        print(fmt.format(name, str(n_ca), str(n_act), str(rmsf), str(e_pen)))


if __name__ == "__main__":
    main()
