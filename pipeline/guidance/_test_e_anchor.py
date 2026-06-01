#!/usr/bin/env python3
"""_test_e_anchor.py - test E_anchor with a synthetic carried anchor injected
into A_cat (proves the term activates correctly when A_cat is enriched).

Scenario: 5OD5-style His-coordinated open leg. Inject a carried anchor at
the local coords of where HIS NE2 should sit. Test on real 5OD5 (expect
reward) and on real 3ZP9 (expect no reward — 3ZP9 has no His at open site).
"""
from __future__ import annotations
import argparse, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guidance.a_cat_fields import ACatFields
from guidance.e_terms.anchor import e_anchor
from guidance._score_pdb import parse_pdb


def inject_carried_anchor(a_cat, pos_local, allowed=("HIS",), donor_atoms=None, sigma=0.8, w=1.5):
    new = json.loads(json.dumps(a_cat))
    new["channels"].setdefault("A_anchor", []).append({
        "type": "synthetic carried anchor (test)",
        "carried": True,
        "pos_local": list(pos_local),
        "allowed_residues": list(allowed),
        "donor_atom_per_res": donor_atoms or {"HIS": ["NE2", "ND1"]},
        "sigma_A": sigma,
        "w": w,
    })
    return new


def find_his_ne2_local(pdb_path, a_cat):
    """Find HIS NE2 closest to the metal in this PDB, return its local pos.
    Used to derive a 'natural' anchor coordinate for the discriminativity test.
    """
    fields = ACatFields(a_cat)
    atoms = parse_pdb(pdb_path)
    best = (None, 1e9)
    for a in atoms:
        if a["resname"] != "HIS": continue
        if a["name"] not in ("NE2", "ND1"): continue
        pl = fields.to_local((a["x"], a["y"], a["z"]))
        d = math.sqrt(sum(c*c for c in pl))
        if d < best[1]:
            best = (pl, d)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a_cat_json")
    ap.add_argument("pdb")
    ap.add_argument("--anchor-pos", nargs=3, type=float, default=[0.0, 0.0, 2.5],
                    help="local coords for synthetic carried anchor (default 0,0,2.5 = coord-shell distance from metal, axial)")
    args = ap.parse_args()

    a = json.load(open(args.a_cat_json))
    pos = tuple(args.anchor_pos)
    print(f"# Anchor pos_local = {pos}  (fixed; tests whether protein has HIS donor at this local coord)")

    atoms = parse_pdb(args.pdb)
    print(f"# A_cat target={a.get('target')}  protein={args.pdb}")

    # Baseline: no carried anchor
    fields0 = ACatFields(a)
    E0 = e_anchor(atoms, fields0)
    print(f"\nbaseline (no carried anchor):           E_anchor = {E0:.4f}")

    # With synthetic anchor at HIS NE2 position
    a_inj = inject_carried_anchor(a, pos)
    fields1 = ACatFields(a_inj)
    E1, per = e_anchor(atoms, fields1, return_per_anchor=True)
    print(f"with carried HIS-NE2 anchor at pos:     E_anchor = {E1:.4f}")
    for pa in per:
        print(f"   -> {pa['type']}: best={pa['best']}  reward={pa['best_reward']}")

    # With wrong allowed residue (only CYS allowed but no CYS near)
    a_inj2 = inject_carried_anchor(a, pos, allowed=("CYS",), donor_atoms={"CYS": ["SG"]})
    fields2 = ACatFields(a_inj2)
    E2 = e_anchor(atoms, fields2)
    print(f"with carried CYS-SG anchor at same pos: E_anchor = {E2:.4f}  (expect ~0; no CYS there)")

    # With shifted anchor (5 A off)
    a_inj3 = inject_carried_anchor(a, [pos[0]+5.0, pos[1], pos[2]])
    fields3 = ACatFields(a_inj3)
    E3 = e_anchor(atoms, fields3)
    print(f"with HIS-NE2 anchor shifted 5 A:        E_anchor = {E3:.4f}  (expect ~0; no HIS there)")


if __name__ == "__main__":
    main()
