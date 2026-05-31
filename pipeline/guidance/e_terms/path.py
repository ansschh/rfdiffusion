#!/usr/bin/env python3
"""E_path - penalize protein matter inside the substrate-approach cone.

Sign convention: HIGHER E_path = MORE blocking of the substrate path = WORSE.

For each protein heavy atom p, contribute:
    e_i = w(atom) * f.eval_path(p)
where eval_path is the soft cone occupancy in [0,1].

E_path(P) = sum_i e_i

Per-atom weight w(atom):
  backbone heavy:   w_bb (default 1.0; backbone occlusion blocks too)
  sidechain heavy:  w_sc (default 1.0)
  hydrogen:         0
  cofactor (LIG):   0   (the cofactor's own ligands obviously occupy the cone -
                          E_path is about protein blocking, not cofactor)

The natural scale: ~5 sidechain heavies fully inside cone -> E_path ~ 5.
A single backbone atom on the cone edge -> E_path ~ 0.5.
"""
from __future__ import annotations
import math
from typing import Iterable

BACKBONE = {"N", "CA", "C", "O", "OXT", "H"}


def e_path(atoms: Iterable[dict], fields,
           w_backbone: float = 1.0, w_sidechain: float = 1.0,
           ignore_resnames=("LIG", "ORI"),
           return_per_atom: bool = False):
    """atoms: iterable of dicts with keys
         {"record", "name", "element", "resname", "chain", "resseq", "x","y","z"}
       fields: ACatFields instance.

    Returns float (E_path) or (E_path, per_atom_list) if return_per_atom.
    """
    total = 0.0
    per = [] if return_per_atom else None
    for a in atoms:
        if a.get("element") == "H": continue
        if a.get("resname") in ignore_resnames: continue
        # Skip explicit metal HETATM (would contribute trivially through path field anyway)
        if a.get("record") == "HETATM" and a.get("resname") not in ("HOH",):
            # Allow water, drop other HETATM (ligands/cofactors)
            continue
        w = w_backbone if a.get("name") in BACKBONE else w_sidechain
        occ = fields.eval_path((a["x"], a["y"], a["z"]))
        e_i = w * occ
        total += e_i
        if per is not None and occ > 1e-3:
            per.append({
                "resname": a["resname"], "chain": a["chain"], "resseq": a["resseq"],
                "name": a["name"], "occ": round(occ, 4), "e_i": round(e_i, 4),
            })
    if return_per_atom:
        return total, per
    return total


def params_doc() -> dict:
    return {
        "term": "E_path",
        "intent": "penalize protein matter inside substrate-approach cone",
        "sign": "higher E_path = more blocking = worse",
        "field_used": "ACatFields.eval_path (soft cone occupancy in [0,1])",
        "atom_filter": {
            "skip_elements": ["H"],
            "skip_resnames": ["LIG", "ORI", "non-water HETATM"],
            "include": "all protein heavy atoms (backbone + sidechain) + waters",
        },
        "weights": {"w_backbone": 1.0, "w_sidechain": 1.0,
                    "note": "Set w_backbone=0 to score sidechain-only blocking."},
        "field_tunables": {
            "path_sigma_perp_A": "0.5 (lateral cone-boundary smoothing)",
            "path_sigma_par_A": "0.5 (axial endpoint smoothing)",
            "path_apex_offset_back_A": "0.0 (V_rxn match would be ~M-H distance, ~1.6 A for 3ZP9)",
            "path_min_extension_past_hydride_A": "0.3 (skip atoms within 0.3 A of hydride)",
        },
        "natural_scale": "5 sidechain heavies fully inside cone -> E_path ~ 5",
    }
