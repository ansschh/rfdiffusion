#!/usr/bin/env python3
"""E_anchor - reward correct anchor residue placement at carried anchor sites.

Sign convention: HIGHER E_anchor = anchor unmet/wrong residue = WORSE.
E_anchor = -sum_a max_r reward(residue, anchor)

An A_cat.A_anchor entry that should contribute looks like:
  {
    "type": "His-coordinated open leg",        # human-readable
    "carried": true,                            # must be true to score
    "pos_local": [x,y,z],                       # where the coord atom should be
    "allowed_residues": ["HIS"],                # which sidechain types satisfy
    "donor_atom_per_res": {"HIS": ["NE2","ND1"]},  # which atom counts (optional)
    "sigma_A": 0.8,                             # Gaussian width
    "w": 1.5,                                   # weight
  }

For v0, the instantiator emits diagnostic-only anchors (carried=false), so
E_anchor returns 0 on existing A_cat data. Once the chem-mode rules or PI
manually populate carried anchors, this term activates.

For 5OD5-style "His-coordinated open leg" cofactors (cp_star_ir_iii_his_ath),
a carried anchor is the natural representation of the chemistry fact that
HIS227 occupies the open metal leg instead of a synthesized hydride.
"""
from __future__ import annotations
import math
from collections import defaultdict

BACKBONE = {"N", "CA", "C", "O", "OXT", "H"}


def _sidechain_centroid_local(residue_atoms, fields):
    sc = [a for a in residue_atoms if a["element"] != "H" and a["name"] not in BACKBONE]
    if not sc:
        ca = next((a for a in residue_atoms if a["name"] == "CA"), None)
        if ca is None: return None
        return fields.to_local((ca["x"], ca["y"], ca["z"]))
    n = len(sc)
    cw = (sum(a["x"] for a in sc)/n, sum(a["y"] for a in sc)/n, sum(a["z"] for a in sc)/n)
    return fields.to_local(cw)


def _named_atom_local(residue_atoms, atom_names, fields, target_pos_local):
    """Return local position of the CLOSEST matching atom (e.g., HIS's
    NE2 vs ND1 — choose whichever is closer to target_pos_local)."""
    best = (None, 1e18)
    for a in residue_atoms:
        if a["name"] not in atom_names: continue
        pl = fields.to_local((a["x"], a["y"], a["z"]))
        d2 = ((pl[0]-target_pos_local[0])**2 +
              (pl[1]-target_pos_local[1])**2 +
              (pl[2]-target_pos_local[2])**2)
        if d2 < best[1]:
            best = (pl, d2)
    return best[0]


def e_anchor(atoms, fields, *,
             return_per_anchor: bool = False):
    """Returns float E_anchor (negative reward) or (E, per_anchor_list)."""
    carried = [a for a in (fields.A_anchor or []) if a.get("carried")]
    if not carried:
        return (0.0, []) if return_per_anchor else 0.0

    by_res = defaultdict(list)
    for a in atoms:
        if a.get("record") != "ATOM": continue
        by_res[(a["chain"], a["resseq"], a["resname"])].append(a)

    per = []
    total_reward = 0.0
    for an in carried:
        pos = an.get("pos_local")
        if pos is None:
            continue
        allowed = set(an.get("allowed_residues", []))
        donor_map = an.get("donor_atom_per_res", {})
        sigma = max(0.3, an.get("sigma_A", 0.8))
        w = an.get("w", 1.0)
        best = (0.0, None)
        for (chain, resseq, resname), ratoms in by_res.items():
            if allowed and resname not in allowed: continue
            # prefer named donor atom if provided, else sidechain centroid
            atom_names = donor_map.get(resname)
            if atom_names:
                rlocal = _named_atom_local(ratoms, atom_names, fields, pos)
            else:
                rlocal = _sidechain_centroid_local(ratoms, fields)
            if rlocal is None: continue
            dx = (rlocal[0]-pos[0])/sigma
            dy = (rlocal[1]-pos[1])/sigma
            dz = (rlocal[2]-pos[2])/sigma
            r = w * math.exp(-0.5*(dx*dx + dy*dy + dz*dz))
            if r > best[0]:
                best = (r, {"chain": chain, "resseq": resseq, "resname": resname, "reward": round(r, 4)})
        total_reward += best[0]
        if return_per_anchor:
            per.append({
                "type": an.get("type"), "pos_local": pos,
                "allowed_residues": list(allowed) or None,
                "best": best[1], "best_reward": round(best[0], 4),
            })
    E = -total_reward
    return (E, per) if return_per_anchor else E


def params_doc():
    return {
        "term": "E_anchor",
        "intent": "reward correct anchor residue (HIS/CYS/etc.) placement at carried anchor sites",
        "sign": "lower (more negative) = better",
        "active_when": "A_cat.A_anchor has at least one entry with carried=true and pos_local set",
        "v0_status": "Returns 0 on current A_cat instantiator output (all anchors diagnostic-only).",
        "scoring": "Per anchor: max over allowed-resname residues of w*exp(-||centroid-pos||^2/(2 sigma^2)). Sum across anchors. Negate.",
        "sigma_default_A": 0.8,
        "w_default": 1.0,
        "donor_atom_specific": "If donor_atom_per_res provided, use named atom (e.g., HIS NE2); else sidechain centroid",
    }
