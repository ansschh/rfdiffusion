#!/usr/bin/env python3
"""_test_e_site.py - discriminativity for E_site.

Test on real 3ZP9 protein, real motif (which has Cp*Ir+N,N+hydride),
synthetic damages: wrong metal (Ir->Zn), missing eta5 (drop 3 Cp* C),
forbidden eta5 (planted in Ru template), planted HIS donor in coord shell.
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guidance.a_cat_fields import ACatFields
from guidance.e_terms.site import e_site
from guidance._score_pdb import parse_pdb


def damage_swap_metal(atoms, new_el="ZN"):
    new = json.loads(json.dumps(atoms))
    for a in new:
        if a.get("record") == "HETATM" and a.get("element") in ("IR", "RH", "RU"):
            a["element"] = new_el
            a["name"] = new_el
            break
    return new


def damage_drop_ring_carbons(atoms, drop_n=3):
    new = []
    dropped = 0
    for a in atoms:
        if (a.get("resname") == "LIG" and a.get("element") == "C"
                and a.get("name", "").startswith("C") and dropped < drop_n):
            dropped += 1
            continue
        new.append(a)
    return new


def damage_plant_his_in_shell(atoms, fields, name="NE2", local=(0.0, 1.0, 2.0)):
    R = fields.R; o = fields.origin
    x = R[0][0]*local[0] + R[1][0]*local[1] + R[2][0]*local[2] + o[0]
    y = R[0][1]*local[0] + R[1][1]*local[1] + R[2][1]*local[2] + o[1]
    z = R[0][2]*local[0] + R[1][2]*local[1] + R[2][2]*local[2] + o[2]
    return atoms + [{"record":"ATOM", "name":name, "element":"N", "resname":"HIS",
                     "chain":"X", "resseq":999, "x":x, "y":y, "z":z}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a_cat_json")
    ap.add_argument("pdb")
    args = ap.parse_args()

    a = json.load(open(args.a_cat_json))
    fields = ACatFields(a)
    atoms = parse_pdb(args.pdb)
    print(f"# A_cat target={a.get('target')}  protein={args.pdb}  n_atoms={len(atoms)}")

    variants = [
        ("real",                              atoms),
        ("wrong metal: Ir -> Zn",             damage_swap_metal(atoms, "ZN")),
        ("missing eta5: drop 3 Cp* carbons",  damage_drop_ring_carbons(atoms, 3)),
        ("planted HIS-NE2 in shell @2A",      damage_plant_his_in_shell(atoms, fields, "NE2", (0.0, 1.0, 2.0))),
        ("planted CYS-SG in shell @2.3A",     damage_plant_his_in_shell(atoms, fields, "SG", (0.0, 1.0, 2.3)) if True else atoms),
    ]
    # also adjust CYS-SG with sg element
    for i, (n, atoms_v) in enumerate(variants):
        if "SG" in n:
            for at in atoms_v:
                if at.get("name") == "SG" and at.get("resname") == "HIS":
                    at["resname"] = "CYS"; at["element"] = "S"

    fmt = "{:<45} {:>10}"
    print()
    print(fmt.format("variant", "E_site"))
    print(fmt.format("-"*45, "-"*10))
    for name, atoms_v in variants:
        E, br = e_site(atoms_v, fields, return_breakdown=True)
        print(fmt.format(name, f"{E:8.3f}"))
        for k, v in br.get("components", {}).items():
            val = v.get("value", 0.0) if isinstance(v, dict) else v
            if abs(val) > 1e-3:
                detail = ""
                if "violations" in v and v["violations"]:
                    detail = f"  {[(x.get('atom',''), x.get('deviation','')) for x in v['violations'][:2]]}"
                if "extras" in v and v["extras"]:
                    detail = f"  {v['extras'][:2]}"
                if "notes" in v and v["notes"]:
                    detail = f"  {v['notes']}"
                print(f"     {k} = {val}{detail}")


if __name__ == "__main__":
    main()
