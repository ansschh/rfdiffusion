#!/usr/bin/env python3
"""_test_heme_failure_mode.py - explicit test of whether the new E_face term
catches the 2CCY heme-transplant failure mode.

The 2CCY failure was: heme has a proximal His that sits on heme's -z axis;
when the candidate-to-query rotation R mapped heme's pocket residues into
3ZP9's frame, heme's proximal His ended up on 3ZP9's +z (reactive face)
because the rotational fit aligned shape but inverted the
proximal/distal handedness.

We simulate this by planting a HIS on 3ZP9's reactive face (+z direction at
~2 A from metal) and computing E_cat under (real A_cat) vs (flipped_face
A_cat). A working chemistry likelihood should:
  - Real A_cat: heavily penalize the planted HIS on reactive face.
  - Flipped A_cat: NOT penalize it (the flipped A_cat thinks +z is packing).

The DIFFERENCE between these two scores is the chemistry-faithfulness signal.
"""
from __future__ import annotations
import argparse, copy, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guidance.a_cat_fields import ACatFields
from guidance.e_cat import e_cat
from guidance.e_terms.face import e_face
from guidance.e_terms.path import e_path
from guidance.damaged_controls import damage_flip_face
from guidance._score_pdb import parse_pdb


def plant_residue(atoms, fields, resname, atoms_local_positions):
    """Add synthetic residue atoms at given local-frame positions.
    atoms_local_positions: dict {atom_name: (x_local, y_local, z_local)}"""
    R = fields.R; o = fields.origin
    new = list(atoms)
    for name, lp in atoms_local_positions.items():
        # world = R^T @ local + origin
        x = R[0][0]*lp[0] + R[1][0]*lp[1] + R[2][0]*lp[2] + o[0]
        y = R[0][1]*lp[0] + R[1][1]*lp[1] + R[2][1]*lp[2] + o[1]
        z = R[0][2]*lp[0] + R[1][2]*lp[1] + R[2][2]*lp[2] + o[2]
        el = name[0] if name[0] in ("N","O","C","S") else "C"
        new.append({"record":"ATOM","name":name,"element":el,"resname":resname,
                    "chain":"X","resseq":999,"x":x,"y":y,"z":z})
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a_cat_json")
    ap.add_argument("pdb")
    args = ap.parse_args()

    a = json.load(open(args.a_cat_json))
    fields_real = ACatFields(a)
    a_flipped = damage_flip_face(json.loads(json.dumps(a)))
    fields_flipped = ACatFields(a_flipped)
    atoms = parse_pdb(args.pdb)

    print(f"# A_cat = {args.a_cat_json}  target={a.get('target')}")
    print(f"# protein = {args.pdb}  n_atoms = {len(atoms)}")
    print()

    # 2CCY heme failure simulation: plant a HIS sidechain with NE2 at +z=2.0
    # (reactive face, coord-shell distance from metal)
    heme_like_his = {
        "CA": (0.0, 0.0, 4.5),     # CA position outside coord shell
        "CB": (0.0, 0.0, 3.5),
        "CG": (0.0, 0.0, 3.0),
        "ND1": (-0.8, 0.0, 2.5),
        "CD2": (0.8, 0.0, 2.5),
        "CE1": (-0.5, 0.0, 1.9),
        "NE2": (0.0, 0.0, 2.0),     # the coordinating nitrogen, AT the open metal site
    }
    atoms_heme = plant_residue(atoms, fields_real, "HIS", heme_like_his)

    # Also plant a benign hydrophobic-packing residue on the -z side (where the
    # natural protein already has hydrophobic packing) for control
    val_packing = {
        "CA": (0.0, 0.0, -4.5),
        "CB": (0.0, 0.0, -3.5),
        "CG1": (0.7, 0.0, -3.0),
        "CG2": (-0.7, 0.0, -3.0),
    }
    atoms_packing = plant_residue(atoms, fields_real, "VAL", val_packing)

    scenarios = [
        ("real 3ZP9 (no planted)",            atoms,         fields_real),
        ("real 3ZP9 + heme-like HIS on +z",   atoms_heme,    fields_real),
        ("real 3ZP9 + benign VAL on -z",      atoms_packing, fields_real),
        ("FLIPPED A_cat | heme-like HIS",     atoms_heme,    fields_flipped),
        ("FLIPPED A_cat | benign VAL on -z",  atoms_packing, fields_flipped),
    ]
    fmt = "{:<40} {:>9} {:>9} {:>10}"
    print(fmt.format("scenario", "E_face", "E_path", "E_cat"))
    print(fmt.format("-"*40, "-"*9, "-"*9, "-"*10))
    for name, at, f in scenarios:
        E, br = e_cat(at, f, return_breakdown=True)
        ef = br["terms"].get("face", 0.0)
        ep = br["terms"].get("path", 0.0)
        print(fmt.format(name, f"{ef:.3f}", f"{ep:.3f}", f"{E:.3f}"))


if __name__ == "__main__":
    main()
