#!/usr/bin/env python3
"""_test_face_on_flipped_protein.py - simulate the 2CCY heme-transplant
failure mode locally by FLIPPING the natural 3ZP9 protein along the z=0 plane
in the cofactor-local frame. The pocket is now on the WRONG handedness side
relative to the cofactor — same geometric failure that the 2CCY transplant
produced when its proximal/distal axis was misaligned.

Expected: E_face should fire on the z-flipped protein (reactive face crowded,
packing face empty), while E_path may NOT fire (residues are still outside
the strict cone, just on the wrong hemisphere).
"""
from __future__ import annotations
import argparse, copy, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guidance.a_cat_fields import ACatFields
from guidance.e_cat import e_cat
from guidance._score_pdb import parse_pdb


def flip_protein_z_in_local_frame(atoms, fields):
    """For each ATOM record (protein), reflect across z=0 in the cofactor
    local frame. HETATM (cofactor) is untouched."""
    R = fields.R; o = fields.origin
    Rt = [[R[0][0], R[1][0], R[2][0]],
          [R[0][1], R[1][1], R[2][1]],
          [R[0][2], R[1][2], R[2][2]]]
    out = []
    for a in atoms:
        if a.get("record") != "ATOM":
            out.append(a); continue
        # to local
        rel = (a["x"]-o[0], a["y"]-o[1], a["z"]-o[2])
        pl = (R[0][0]*rel[0]+R[0][1]*rel[1]+R[0][2]*rel[2],
              R[1][0]*rel[0]+R[1][1]*rel[1]+R[1][2]*rel[2],
              R[2][0]*rel[0]+R[2][1]*rel[1]+R[2][2]*rel[2])
        # flip z
        pl_flipped = (pl[0], pl[1], -pl[2])
        # back to world
        wx = Rt[0][0]*pl_flipped[0]+Rt[0][1]*pl_flipped[1]+Rt[0][2]*pl_flipped[2] + o[0]
        wy = Rt[1][0]*pl_flipped[0]+Rt[1][1]*pl_flipped[1]+Rt[1][2]*pl_flipped[2] + o[1]
        wz = Rt[2][0]*pl_flipped[0]+Rt[2][1]*pl_flipped[1]+Rt[2][2]*pl_flipped[2] + o[2]
        an = copy.copy(a); an["x"] = wx; an["y"] = wy; an["z"] = wz
        out.append(an)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a_cat_json")
    ap.add_argument("pdb")
    args = ap.parse_args()

    a = json.load(open(args.a_cat_json))
    fields = ACatFields(a)
    atoms_real = parse_pdb(args.pdb)
    atoms_flipped = flip_protein_z_in_local_frame(atoms_real, fields)

    print(f"# A_cat = {args.a_cat_json}  target={a.get('target')}")
    print(f"# This test reflects the protein across z=0 in cofactor-local frame.")
    print(f"# The cofactor stays put; the pocket moves to the wrong handedness side.")
    print(f"# Simulates the 2CCY heme failure mode: protein on wrong proximal/distal side.")
    print()

    fmt = "{:<35} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10}"
    print(fmt.format("scenario", "E_path", "E_contact", "E_avoid", "E_face", "E_cz", "E_site", "E_cat"))
    print(fmt.format("-"*35, "-"*9, "-"*9, "-"*9, "-"*9, "-"*9, "-"*9, "-"*10))
    for name, at in [("real 3ZP9", atoms_real),
                      ("3ZP9 protein z-flipped (heme-like)", atoms_flipped)]:
        E, br = e_cat(at, fields, return_breakdown=True)
        t = br["terms"]
        print(fmt.format(name,
                          f"{t.get('path',0):.3f}",
                          f"{t.get('contact',0):.3f}",
                          f"{t.get('avoid',0):.3f}",
                          f"{t.get('face',0):.3f}",
                          f"{t.get('coord_zones',0):.3f}",
                          f"{t.get('site',0):.3f}",
                          f"{E:.3f}"))


if __name__ == "__main__":
    main()
