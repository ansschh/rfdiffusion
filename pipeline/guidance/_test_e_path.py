#!/usr/bin/env python3
"""_test_e_path.py - quick discriminativity probe for E_path before formal
damaged-control generator. Score the same protein under real A_cat and under
inline-damaged variants (axis flipped, axis rotated, cone widened).

Usage:
  python _test_e_path.py pipeline/compiled/3ZP9/A_cat.json pipeline/pdb/3ZP9.pdb
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guidance.a_cat_fields import ACatFields
from guidance.e_terms.path import e_path
from guidance._score_pdb import parse_pdb


def make_variant(a_cat, **path_overrides):
    """Return a copy of a_cat with A_path tweaked per overrides."""
    new = json.loads(json.dumps(a_cat))
    p = new["channels"]["A_path"]
    for k, v in path_overrides.items():
        p[k] = v
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a_cat_json")
    ap.add_argument("pdb")
    ap.add_argument("--apex-back", type=float, default=0.0)
    args = ap.parse_args()

    a = json.load(open(args.a_cat_json))
    atoms = parse_pdb(args.pdb)
    print(f"# A_cat={args.a_cat_json}  target={a.get('target')}  mode={a.get('mode')}")
    print(f"# Protein={args.pdb}  n_atoms={len(atoms)}  apex_back={args.apex_back}")
    print()

    variants = [
        ("real (axis = +z)",            {"axis_local": [0.0, 0.0, 1.0]}),
        ("axis flipped (-z, into M)",   {"axis_local": [0.0, 0.0, -1.0]}),
        ("axis +x (lateral)",           {"axis_local": [1.0, 0.0, 0.0]}),
        ("axis -x (lateral)",           {"axis_local": [-1.0, 0.0, 0.0]}),
        ("axis +y (lateral)",           {"axis_local": [0.0, 1.0, 0.0]}),
        ("axis -y (lateral)",           {"axis_local": [0.0, -1.0, 0.0]}),
        ("axis 45 (xz quadrant)",       {"axis_local": [0.707, 0.0, 0.707]}),
        ("axis 135 (xz quadrant)",      {"axis_local": [-0.707, 0.0, 0.707]}),
        ("widened cone 60 deg",         {"half_angle_deg": 60.0}),
        ("doubled extent 11 A",         {"extent_A": 11.0}),
    ]
    fmt = "{:<32} {:>10} {:>10} {:>10}"
    print(fmt.format("variant", "E_path", "sc-only", "bb-only"))
    print(fmt.format("-"*32, "-"*10, "-"*10, "-"*10))
    for name, overrides in variants:
        a_var = make_variant(a, **overrides)
        f = ACatFields(a_var, path_apex_offset_back=args.apex_back)
        E_full = e_path(atoms, f, w_backbone=1.0, w_sidechain=1.0)
        E_sc   = e_path(atoms, f, w_backbone=0.0, w_sidechain=1.0)
        E_bb   = e_path(atoms, f, w_backbone=1.0, w_sidechain=0.0)
        print(fmt.format(name, f"{E_full:8.3f}", f"{E_sc:8.3f}", f"{E_bb:8.3f}"))


if __name__ == "__main__":
    main()
