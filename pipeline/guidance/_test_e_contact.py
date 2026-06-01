#!/usr/bin/env python3
"""_test_e_contact.py - quick discriminativity probe for E_contact.

Score the same protein under real A_cat.A_contact and under inline-damaged
variants (Gaussians flipped to wrong side, types shuffled to wrong classes,
positions randomized).
"""
from __future__ import annotations
import argparse, json, math, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guidance.a_cat_fields import ACatFields
from guidance.e_terms.contact import e_contact, RESIDUE_TYPE
from guidance._score_pdb import parse_pdb


def damage_flip(a_cat):
    """Flip mu_local across z=0 plane (move Gaussians to the substrate-cone side)."""
    new = json.loads(json.dumps(a_cat))
    for g in new["channels"]["A_contact"]:
        g["mu_local"][2] = -g["mu_local"][2]
    return new


def damage_swap_types(a_cat):
    """Cycle types so each Gaussian asks for the wrong residue class."""
    new = json.loads(json.dumps(a_cat))
    cycle = {"hydrophobic": "charged_acid", "aromatic": "charged_base",
             "polar": "hydrophobic", "anchor": "hydrophobic",
             "charged_acid": "hydrophobic", "charged_base": "hydrophobic"}
    for g in new["channels"]["A_contact"]:
        g["type"] = cycle.get(g.get("type"), "small")
    return new


def damage_randomize_positions(a_cat, seed=42, span=8.0):
    new = json.loads(json.dumps(a_cat))
    rng = random.Random(seed)
    for g in new["channels"]["A_contact"]:
        g["mu_local"] = [round(rng.uniform(-span, span), 3) for _ in range(3)]
    return new


def damage_rotate_about_z(a_cat, deg):
    new = json.loads(json.dumps(a_cat))
    th = math.radians(deg); c = math.cos(th); s = math.sin(th)
    for g in new["channels"]["A_contact"]:
        x, y, z = g["mu_local"]
        g["mu_local"] = [round(c*x - s*y, 3), round(s*x + c*y, 3), z]
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a_cat_json")
    ap.add_argument("pdb")
    ap.add_argument("--agg", choices=["winner_take_all", "sum"], default="winner_take_all")
    args = ap.parse_args()

    a = json.load(open(args.a_cat_json))
    atoms = parse_pdb(args.pdb)
    print(f"# A_cat={args.a_cat_json}  target={a.get('target')}  mode={a.get('mode')}")
    print(f"# Protein={args.pdb}  n_atoms={len(atoms)}  agg={args.agg}")
    print()

    variants = [
        ("real",                          a),
        ("flip z (substrate side)",       damage_flip(a)),
        ("swap types",                    damage_swap_types(a)),
        ("rotate 90 about z",             damage_rotate_about_z(a, 90)),
        ("rotate 180 about z",            damage_rotate_about_z(a, 180)),
        ("randomize positions (seed 42)", damage_randomize_positions(a, 42)),
        ("randomize positions (seed 7)",  damage_randomize_positions(a, 7)),
        ("randomize positions (seed 13)", damage_randomize_positions(a, 13)),
    ]
    fmt = "{:<34} {:>12}  {:>10}"
    print(fmt.format("variant", "E_contact", "n_matched"))
    print(fmt.format("-"*34, "-"*12, "-"*10))
    for name, a_var in variants:
        f = ACatFields(a_var)
        E, per = e_contact(atoms, f, agg=args.agg, return_per_gaussian=True)
        n_matched = sum(1 for p in per if p["best_reward"] > 0.05)
        print(fmt.format(name, f"{E:8.3f}", f"{n_matched}/{len(per)}"))
        if name == "real":
            for p in per:
                print(f"     -> [{p['type']:>12}] mu={p['mu_local']} reward={p['best_reward']} best={p['best_residue']} (src={p['source']})")


if __name__ == "__main__":
    main()
