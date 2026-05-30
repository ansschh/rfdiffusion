#!/usr/bin/env python3
"""transplant_pocket.py — take a retrieved heme/other pocket and map its residues into the
query target's local frame, ready to be compiled into a new RFD2 motif as 'transplanted
guideposts'. Tests the PI's question: can local redesign of a top-retrieved non-target pocket
pass V_chem + V_rxn for the query's chemistry?

Usage:
  python transplant_pocket.py \
    --query-acat pipeline/compiled/3ZP9/A_cat.json \
    --query-target 3ZP9 \
    --best-R pipeline/retrieval/best_R.json \
    --candidate-pdb pipeline/retrieval/library/pdb/2CCY.pdb \
    --candidate-id 2CCY \
    --top-residues 4 \
    --out pipeline/retrieval/transplant/3ZP9_from_2CCY.json
"""
from __future__ import annotations
import argparse, json, math, os
from collections import defaultdict

METALS = {"IR","ZN","RH","RU","FE","MN","CU","CO","NI","PD","PT","MO","W","OS","V","CR","MG","CA","NA","K","AL"}
BACKBONE = {"N","CA","C","O","OXT","H"}


def parse_pdb(path):
    out = []
    for line in open(path):
        if line[:6].strip() not in ("ATOM","HETATM"): continue
        try:
            name = line[12:16].strip()
            el = line[76:78].strip().upper() or "".join(c for c in name if c.isalpha())[:2].upper()
            out.append({"record": line[:6].strip(), "name": name, "element": el,
                        "resname": line[17:20].strip(), "chain": line[21].strip(),
                        "resseq": int(line[22:26]),
                        "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54])})
        except ValueError: continue
    return out


def matmul3_vec(R, v):
    return (R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2],
            R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2],
            R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2])


def transpose3(R):
    return [[R[0][0], R[1][0], R[2][0]],
            [R[0][1], R[1][1], R[2][1]],
            [R[0][2], R[1][2], R[2][2]]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-acat", required=True, help="A_cat JSON for the query target")
    ap.add_argument("--best-R", required=True, help="best_R JSON from retrieve.py --save-best-R")
    ap.add_argument("--candidate-pdb", required=True, help="PDB file for the candidate pocket")
    ap.add_argument("--candidate-id", required=True, help="PDB id key into best_R JSON (e.g., 2CCY)")
    ap.add_argument("--top-residues", type=int, default=4, help="N closest-to-metal residues to transplant")
    ap.add_argument("--max-residue-dist", type=float, default=8.0,
                    help="ignore candidate residues farther than this from candidate metal (A)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    query = json.load(open(args.query_acat))
    best_R_data = json.load(open(args.best_R))
    if args.candidate_id not in best_R_data:
        raise SystemExit(f"candidate id '{args.candidate_id}' not in best_R JSON; available: {list(best_R_data)[:10]}")
    cand_info = best_R_data[args.candidate_id]
    R = cand_info["best_R"]                       # maps world_cand -> local_query (v_local = R @ (v_world - metal_world_cand))
    R_T = transpose3(R)                           # inverse rotation (R is orthogonal)
    metal_cand_world = tuple(cand_info["metal_world"])
    metal_query_world = tuple(query["frame"]["origin_world"])
    R_query = query["frame"]["R_world_to_local"]   # rows = query-local axes in world frame
    R_query_T = transpose3(R_query)

    atoms = parse_pdb(args.candidate_pdb)
    # Find candidate metal (sanity: should be at metal_cand_world)
    metals = [a for a in atoms if a["element"] in METALS]
    if not metals:
        raise SystemExit(f"no metal in {args.candidate_pdb}")

    # Group atoms by residue (ATOM records only — protein side)
    by_res = defaultdict(list)
    for a in atoms:
        if a["record"] == "ATOM":
            by_res[(a["chain"], a["resseq"])].append(a)

    # Compute each residue's min-dist-to-candidate-metal (sidechain heavies only)
    candidates = []
    for (chain, resseq), res_atoms in by_res.items():
        resname = res_atoms[0]["resname"]
        sc = [a for a in res_atoms if a["element"] != "H" and a["name"] not in BACKBONE]
        if not sc:
            continue
        min_d = min(math.sqrt((a["x"]-metal_cand_world[0])**2 + (a["y"]-metal_cand_world[1])**2 +
                              (a["z"]-metal_cand_world[2])**2) for a in sc)
        if min_d > args.max_residue_dist:
            continue
        candidates.append({"chain": chain, "resseq": resseq, "resname": resname,
                           "min_dist": min_d, "all_heavies": [a for a in res_atoms if a["element"] != "H"]})
    candidates.sort(key=lambda r: r["min_dist"])
    chosen = candidates[:args.top_residues]
    if not chosen:
        raise SystemExit("no residues found within max-residue-dist of candidate metal")

    # Transform each residue's atoms: candidate world -> query local -> query world
    transplanted = []
    for i, res in enumerate(chosen):
        new_atoms = []
        for a in res["all_heavies"]:
            v_world_cand = (a["x"]-metal_cand_world[0], a["y"]-metal_cand_world[1], a["z"]-metal_cand_world[2])
            v_local = matmul3_vec(R, v_world_cand)
            v_world_query = matmul3_vec(R_query_T, v_local)
            new_atoms.append({"name": a["name"], "element": a["element"],
                              "x": round(v_world_query[0] + metal_query_world[0], 3),
                              "y": round(v_world_query[1] + metal_query_world[1], 3),
                              "z": round(v_world_query[2] + metal_query_world[2], 3)})
        # use chain 'A' and resseq 900+i for transplanted residues (avoid collision with native numbering)
        transplanted.append({
            "chain": "A",
            "resseq": 900 + i,
            "resname": res["resname"],
            "source_pdb": args.candidate_id,
            "source_residue": f"{res['resname']}{res['resseq']}{res['chain']}",
            "min_dist_to_candidate_metal": round(res["min_dist"], 3),
            "atoms": new_atoms,
        })

    out = {
        "query_target": query["target"],
        "candidate_source": args.candidate_id,
        "transplanted_guideposts": transplanted,
        "note": "Coordinates expressed in QUERY world frame (origin = query metal); ready for "
                "motif_compiler.py --external-guideposts.",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"=== transplanted {len(transplanted)} residues: {args.candidate_id} -> {query['target']} frame ===")
    for r in transplanted:
        print(f"  {r['resname']}{r['resseq']}{r['chain']}  source={r['source_residue']}  d={r['min_dist_to_candidate_metal']} A")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
