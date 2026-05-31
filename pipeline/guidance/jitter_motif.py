#!/usr/bin/env python3
"""jitter_motif.py - create a perturbed motif.pdb by jittering guidepost
positions. Used by Stage 1 outer-loop SMC: each particle gets a slightly
different motif, and the importance-resampling step concentrates on motif
configurations that produce high-likelihood designs.

Jitter scheme:
  For each guidepost residue, sample a random translation t ~ N(0, sigma^2 I_3),
  apply to ALL atoms of that residue (so the residue keeps its internal geometry).
  Cofactor atoms (LIG / ORI) are NEVER jittered - the rigid chemistry stays
  fixed.

Optional seeded operation for reproducibility / SMC bookkeeping.
"""
from __future__ import annotations
import argparse, json, os, random
from typing import List


def parse_pdb_lines(path):
    return open(path).readlines()


def write_pdb(lines, path):
    with open(path, "w") as f:
        f.writelines(lines)


def jitter_pdb(in_path: str, out_path: str, sigma_A: float = 1.0,
               seed: int = 0, jitter_chains_resseqs: List = None,
               fixed_resnames=("LIG", "ORI")):
    """Jitter guidepost residues. If jitter_chains_resseqs is None, jitters
    all ATOM records (preserving each residue's internal geometry). HETATM
    records with resname in fixed_resnames are NEVER moved.
    """
    rng = random.Random(seed)
    lines = parse_pdb_lines(in_path)
    # group ATOM records by (chain, resseq)
    res_translations = {}     # (chain, resseq) -> (dx, dy, dz)
    out = []
    for line in lines:
        rec = line[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            out.append(line); continue
        resname = line[17:20].strip()
        if rec == "HETATM" and resname in fixed_resnames:
            out.append(line); continue
        try:
            chain = line[21].strip()
            resseq = int(line[22:26])
        except ValueError:
            out.append(line); continue
        if jitter_chains_resseqs is not None:
            if (chain, resseq) not in jitter_chains_resseqs:
                out.append(line); continue
        key = (chain, resseq)
        if key not in res_translations:
            res_translations[key] = (rng.gauss(0, sigma_A),
                                     rng.gauss(0, sigma_A),
                                     rng.gauss(0, sigma_A))
        dx, dy, dz = res_translations[key]
        try:
            x = float(line[30:38]) + dx
            y = float(line[38:46]) + dy
            z = float(line[46:54]) + dz
            new = line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
            out.append(new)
        except ValueError:
            out.append(line)
    write_pdb(out, out_path)
    return res_translations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motif_pdb")
    ap.add_argument("out_pdb")
    ap.add_argument("--sigma", type=float, default=1.0, help="jitter stddev in A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--residues", default=None,
                    help="comma-separated <chain><resseq> list, e.g. A131,A135,A198,A202")
    args = ap.parse_args()
    sel = None
    if args.residues:
        sel = set()
        for tok in args.residues.split(","):
            tok = tok.strip()
            if not tok: continue
            chain = tok[0]
            resseq = int(tok[1:])
            sel.add((chain, resseq))
    translations = jitter_pdb(args.motif_pdb, args.out_pdb, sigma_A=args.sigma,
                              seed=args.seed, jitter_chains_resseqs=sel)
    print(f"# jittered {len(translations)} residue(s) (sigma={args.sigma} A, seed={args.seed})")
    for (ch, rs), t in sorted(translations.items()):
        mag = (t[0]**2 + t[1]**2 + t[2]**2) ** 0.5
        print(f"   {ch}{rs}  dx={t[0]:+.3f}  dy={t[1]:+.3f}  dz={t[2]:+.3f}  |t|={mag:.3f} A")
    print(f"# wrote {args.out_pdb}")


if __name__ == "__main__":
    main()
