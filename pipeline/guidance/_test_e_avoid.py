#!/usr/bin/env python3
"""_test_e_avoid.py - discriminativity probe for E_avoid.

Score real 3ZP9 vs synthetic "mutations" where we plant a HIS/CYS near the
metal coordination shell to fake a donor-poisoning event.
"""
from __future__ import annotations
import argparse, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guidance.a_cat_fields import ACatFields
from guidance.e_terms.avoid import e_avoid
from guidance._score_pdb import parse_pdb


def plant_donor_at_local(atoms, fields, donor_atom_name, resname, local_pos):
    """Plant a single donor atom at the given local coords (will be transformed
    to world for inclusion in the atom list).
    """
    R = fields.R
    origin = fields.origin
    # world = R^T @ local + origin   (R rows = local axes in world)
    x = R[0][0]*local_pos[0] + R[1][0]*local_pos[1] + R[2][0]*local_pos[2] + origin[0]
    y = R[0][1]*local_pos[0] + R[1][1]*local_pos[1] + R[2][1]*local_pos[2] + origin[1]
    z = R[0][2]*local_pos[0] + R[1][2]*local_pos[1] + R[2][2]*local_pos[2] + origin[2]
    el = "S" if donor_atom_name == "SG" else ("N" if donor_atom_name.startswith("N") else "O")
    return atoms + [{
        "record": "ATOM", "name": donor_atom_name, "element": el,
        "resname": resname, "chain": "X", "resseq": 999,
        "x": x, "y": y, "z": z,
    }]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a_cat_json")
    ap.add_argument("pdb")
    args = ap.parse_args()

    a = json.load(open(args.a_cat_json))
    fields = ACatFields(a)
    atoms = parse_pdb(args.pdb)
    print(f"# A_cat={args.a_cat_json}  target={a.get('target')}  mode={a.get('mode')}")
    print(f"# Protein={args.pdb}  n_atoms={len(atoms)}")
    print()

    variants = [
        ("real (no planted donor)",          atoms),
        ("planted HIS-ND1 at d=2.0 A z=2",   plant_donor_at_local(atoms, fields, "ND1", "HIS", [0.0, 0.0, 2.0])),
        ("planted HIS-ND1 at d=2.5 A z=2.5", plant_donor_at_local(atoms, fields, "ND1", "HIS", [0.0, 0.0, 2.5])),
        ("planted HIS-ND1 at d=3.5 A",       plant_donor_at_local(atoms, fields, "ND1", "HIS", [0.0, 0.0, 3.5])),
        ("planted CYS-SG at d=2.0 A z=2",    plant_donor_at_local(atoms, fields, "SG",  "CYS", [0.0, 0.0, 2.0])),
        ("planted ASP-OD1 at d=2.0 A z=2",   plant_donor_at_local(atoms, fields, "OD1", "ASP", [0.0, 0.0, 2.0])),
        ("planted SER-OG at d=2.0 A",        plant_donor_at_local(atoms, fields, "OG",  "SER", [0.0, 0.0, 2.0])),
        ("planted HIS-ND1 lateral d=2.0",    plant_donor_at_local(atoms, fields, "ND1", "HIS", [2.0, 0.0, 0.0])),
    ]
    fmt = "{:<38} {:>10} {:>10} {:>10} {:>10}"
    print(fmt.format("variant", "E_avoid", "E_donor", "E_steric", "n_donors"))
    print(fmt.format("-"*38, "-"*10, "-"*10, "-"*10, "-"*10))
    for name, atom_list in variants:
        E, comp = e_avoid(atom_list, fields, return_components=True)
        print(fmt.format(name, f"{E:8.3f}", f"{comp['E_donor']:8.3f}", f"{comp['E_steric']:8.3f}", str(comp["n_donors_near_metal"])))
        if "planted" in name and comp["top_donors"]:
            for d in comp["top_donors"][:2]:
                print(f"     -> {d['resname']}{d['resseq']}{d['chain']}/{d['atom']} d={d['d_metal']} field={d['field']} contrib={d['contrib']}")
    print()
    print("# Top donors in REAL protein (no planted donor):")
    E, comp = e_avoid(atoms, fields, return_components=True)
    for d in comp["top_donors"][:8]:
        print(f"  {d['resname']}{d['resseq']}{d['chain']}/{d['atom']} d={d['d_metal']} field={d['field']} contrib={d['contrib']}")
    print(f"# Top steric clashes in REAL protein:")
    for s in comp["top_steric"][:5]:
        print(f"  {s['resname']}{s['resseq']}{s['chain']}/{s['atom']} field={s['field']} contrib={s['contrib']}")


if __name__ == "__main__":
    main()
