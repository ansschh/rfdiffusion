#!/usr/bin/env python3
"""_test_e_seq_chem.py - discriminativity for E_seq_chem.

Plant Cys/Met/His near metal vs same residue identity elsewhere.
"""
from __future__ import annotations
import argparse, json, os, sys, math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guidance.a_cat_fields import ACatFields
from guidance.e_terms.seq_chem import e_seq_chem
from guidance._score_pdb import parse_pdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a_cat_json")
    ap.add_argument("pdb")
    args = ap.parse_args()

    a = json.load(open(args.a_cat_json))
    fields = ACatFields(a)
    atoms = parse_pdb(args.pdb)
    print(f"# A_cat target={a.get('target')}  protein={args.pdb}  metal_class={('soft' if any(c.get('element','').upper() in {'IR','RH','RU','PT','PD','OS'} for c in a.get('cofactor_atoms_local',[])) else 'hard')}")

    E, info = e_seq_chem(atoms, fields, return_per_residue=True)
    print(f"\nReal protein:  E_seq_chem = {E:.4f}  (E_pen={info['E_penalty']}, E_rew={info['E_reward']})")
    print(f"               metal_class={info['metal_class']}, cationic_ts={info['cationic_ts']}")
    print(f"Top contributing residues:")
    for r in info["per_residue"][:10]:
        print(f"   {r['resname']}{r['resseq']}{r['chain']} d={r['d_metal']}A net={r['net']}  reasons={r['reasons']}")

    # Mutate: rename a few PHE/LEU near metal to CYS to fake a poisoning mutation
    # 3ZP9 PHE131 and VAL135 are the closest hydrophobic residues. Mutate one to CYS.
    print(f"\n--- Synthetic mutation test ---")
    for target_res in ("PHE", "VAL", "LEU"):
        for (chain, resseq), _ in [(("A", 131), None), (("A", 135), None)]:
            mutated = json.loads(json.dumps(atoms))
            n_mut = 0
            for a_atom in mutated:
                if a_atom["chain"] == chain and a_atom["resseq"] == resseq and a_atom["resname"] == target_res:
                    a_atom["resname"] = "CYS"
                    n_mut += 1
            if n_mut > 0:
                E2 = e_seq_chem(mutated, fields)
                # also try MET, HIS
                for to in ("CYS", "MET", "HIS"):
                    mutated2 = json.loads(json.dumps(atoms))
                    for a_atom in mutated2:
                        if a_atom["chain"] == chain and a_atom["resseq"] == resseq and a_atom["resname"] == target_res:
                            a_atom["resname"] = to
                    E_to = e_seq_chem(mutated2, fields)
                    print(f"  {target_res}{resseq}{chain} -> {to}: E_seq_chem = {E_to:.4f}  (delta {E_to - E:.4f})")
                break


if __name__ == "__main__":
    main()
