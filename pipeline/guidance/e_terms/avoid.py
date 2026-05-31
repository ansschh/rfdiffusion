#!/usr/bin/env python3
"""E_avoid - penalize protein matter in catalyst-poisoning regions.

Sign convention: HIGHER E_avoid = MORE poisoning = WORSE.

Two components in v0:

  (a) Donor poisoning - protein sidechain donor atoms inside the metal's
      coordination shell. ANY donor atom there competes with cofactor donors
      and poisons the active site. The eval_avoid field is a Gaussian peak
      at radius avoid_radius_around_metal (default 2.0 A), width avoid_sigma
      (default 0.5 A). Weighted by donor_strength (S, N-imidazole > N-amide,
      carboxylate > hydroxyl).

  (b) Steric shell - extra protein heavy atoms inside the A_steric extended
      vdW shells of cofactor atoms. Reuses ACatFields.eval_steric (sigmoid
      around vdW+0.4). Backbone atoms penalized at half weight (designable
      around) vs sidechain (which we should be able to place anywhere).

E_avoid(P) = w_donor * sum(donor_poison) + w_steric * sum(steric_clash)

Natural scale: a single HIS imidazole 2 A from the metal -> E_donor ~ 1.5;
a single backbone CA inside cofactor vdW shell -> E_steric ~ 0.5.
"""
from __future__ import annotations
import math


# Sidechain donor atom names by residue. Backbone N/O excluded (those are
# everywhere in a folded chain; the metal-binding test is sidechain donors).
DONOR_ATOMS = {
    "HIS": {"ND1": 1.5, "NE2": 1.5},
    "ASP": {"OD1": 1.0, "OD2": 1.0},
    "GLU": {"OE1": 1.0, "OE2": 1.0},
    "SER": {"OG":  0.5},
    "THR": {"OG1": 0.5},
    "TYR": {"OH":  0.7},     # tyrosinate
    "ASN": {"OD1": 0.4, "ND2": 0.3},
    "GLN": {"OE1": 0.4, "NE2": 0.3},
    "CYS": {"SG":  2.0},     # strongest donor; poisons soft metals
    "MET": {"SD":  1.2},     # soft donor
    "LYS": {"NZ":  0.8},
    "ARG": {"NH1": 0.5, "NH2": 0.5, "NE": 0.3},
    "TRP": {"NE1": 0.4},
}

BACKBONE = {"N", "CA", "C", "O", "OXT", "H"}


def e_avoid(atoms, fields, *,
            w_donor: float = 1.0, w_steric: float = 1.0,
            steric_w_backbone: float = 0.5, steric_w_sidechain: float = 1.0,
            max_donor_check_dist_A: float = 5.0,
            return_components: bool = False):
    """atoms: iterable of atom dicts. fields: ACatFields.

    Returns float E_avoid, or (E, components_dict) if return_components.
    """
    e_donor = 0.0
    donors_found = []
    e_steric = 0.0
    steric_hits = []

    for a in atoms:
        if a.get("element") == "H": continue
        rec = a.get("record")
        nm = a.get("name")
        resname = a.get("resname")

        # (a) donor poisoning - ATOM records only, sidechain donor names
        if rec == "ATOM":
            strength = DONOR_ATOMS.get(resname, {}).get(nm)
            if strength is not None:
                p = (a["x"], a["y"], a["z"])
                pl = fields.to_local(p)
                d = math.sqrt(pl[0]*pl[0] + pl[1]*pl[1] + pl[2]*pl[2])
                if d <= max_donor_check_dist_A:
                    penalty = fields.eval_avoid(p)   # Gaussian peak at avoid_radius
                    contrib = w_donor * strength * penalty
                    e_donor += contrib
                    if contrib > 0.05:
                        donors_found.append({
                            "resname": resname, "chain": a["chain"], "resseq": a["resseq"],
                            "atom": nm, "d_metal": round(d, 3),
                            "donor_strength": strength, "field": round(penalty, 4),
                            "contrib": round(contrib, 4),
                        })

        # (b) steric shell - all heavy atoms (ATOM or HETATM water); skip cofactor
        if resname in ("LIG", "ORI"): continue
        if rec == "HETATM" and resname not in ("HOH",): continue
        w_st = steric_w_backbone if nm in BACKBONE else steric_w_sidechain
        st = fields.eval_steric((a["x"], a["y"], a["z"]))
        contrib = w_steric * w_st * st
        e_steric += contrib
        if contrib > 0.1:
            steric_hits.append({
                "resname": resname, "chain": a.get("chain"), "resseq": a.get("resseq"),
                "atom": nm, "field": round(st, 4), "contrib": round(contrib, 4),
            })

    E = e_donor + e_steric
    if return_components:
        return E, {
            "E_donor": round(e_donor, 4),
            "E_steric": round(e_steric, 4),
            "n_donors_near_metal": len(donors_found),
            "n_steric_hits": len(steric_hits),
            "top_donors": sorted(donors_found, key=lambda x: -x["contrib"])[:5],
            "top_steric": sorted(steric_hits, key=lambda x: -x["contrib"])[:5],
        }
    return E


def params_doc():
    return {
        "term": "E_avoid",
        "intent": "penalize protein donor atoms inside metal coordination shell + extra steric clashes",
        "sign": "higher E_avoid = more poisoning/clash = worse",
        "components": {
            "E_donor": {
                "atoms": "sidechain donor atoms only (DONOR_ATOMS table)",
                "field": "ACatFields.eval_avoid - Gaussian peak at avoid_radius_around_metal=2.0 A, sigma=0.5",
                "donor_strength_scale": "CYS 2.0 > MET 1.5 > HIS 1.5 > ASP/GLU 1.0 > LYS 0.8 > TYR 0.7 > SER/THR 0.5 > ASN/GLN/ARG 0.3-0.5",
                "weight": "w_donor = 1.0",
                "natural_scale": "HIS imidazole at 2 A from metal -> ~1.5",
            },
            "E_steric": {
                "atoms": "all protein heavy atoms (backbone + sidechain), waters",
                "field": "ACatFields.eval_steric - sigmoid around (vdW+0.4) of each cofactor atom",
                "weights": "w_backbone=0.5 (designable around), w_sidechain=1.0",
                "natural_scale": "single CA inside vdW+0.4 shell -> ~0.5",
            },
        },
    }
