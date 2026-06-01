#!/usr/bin/env python3
"""E_dynamics_proxy - CHEAP "static sculpture" detector.

This is NOT a real dynamics measurement. Full V_preorg requires PyRosetta /
OpenMM relax + normal modes + short MD. That work belongs in a separate
post-design validator (V_preorg), not in the in-denoiser SMC loop.

What this term IS: a per-design proxy that flags pockets whose backbone
geometry around the active site is irregular enough to suggest the design
is a "static sculpture" — a structurally feasible pocket that wouldn't
survive minimization.

Sign convention: HIGHER E_dynamics_proxy = MORE irregular pocket geometry = WORSE.

Components:
  1. BB-NEIGHBOR variance: For active-site residues (CA within radius of metal),
     compute pairwise CA-CA distances to N nearest other active-site CAs.
     Standard deviation of these distances. Low = ordered, high = irregular.
  2. SS deficit: penalty if too few active-site residues are in canonical
     secondary structure (estimated from CA-CA local geometry: alpha-helix
     i,i+3 distance ~5.3 A; beta-strand i,i+1 distance ~3.5 A with extended
     phi/psi). Very rough.

DRAFT proxy. Calibration needed. The PI's "static sculpture" intuition is
real; this is a cheap stand-in until the post-design V_preorg lands.
"""
from __future__ import annotations
import math
from collections import defaultdict


def _ca_atoms_active_site(atoms, fields, radius_A: float = 8.0):
    """Collect (resseq, chain, resname, world_xyz, local_xyz) for protein CAs
    within `radius_A` of the metal (origin in local frame)."""
    out = []
    for a in atoms:
        if a.get("record") != "ATOM": continue
        if a.get("name") != "CA": continue
        p = (a["x"], a["y"], a["z"])
        pl = fields.to_local(p)
        d = math.sqrt(pl[0]*pl[0] + pl[1]*pl[1] + pl[2]*pl[2])
        if d > radius_A: continue
        out.append({
            "chain": a["chain"], "resseq": a["resseq"], "resname": a["resname"],
            "world": p, "local": pl, "d_metal": d,
        })
    return out


def _pairwise_dists(world_coords):
    n = len(world_coords)
    out = []
    for i in range(n):
        for j in range(i+1, n):
            wi, wj = world_coords[i], world_coords[j]
            d = math.sqrt((wi[0]-wj[0])**2 + (wi[1]-wj[1])**2 + (wi[2]-wj[2])**2)
            out.append(d)
    return out


def _bb_neighbor_variance(active_cas, k_neighbors: int = 3):
    """For each active-site CA, average distance to its k nearest other
    active-site CAs. Return the stddev of those average-neighbor-distances
    across all active-site CAs. Low variance = ordered packing."""
    if len(active_cas) < k_neighbors + 1:
        return None
    means = []
    for i, ca_i in enumerate(active_cas):
        dists_to_others = []
        for j, ca_j in enumerate(active_cas):
            if i == j: continue
            d = math.sqrt(sum((ca_i["world"][k]-ca_j["world"][k])**2 for k in range(3)))
            dists_to_others.append(d)
        dists_to_others.sort()
        # average of k nearest
        if len(dists_to_others) < k_neighbors: continue
        means.append(sum(dists_to_others[:k_neighbors]) / k_neighbors)
    if len(means) < 2:
        return None
    mean = sum(means) / len(means)
    var = sum((m-mean)**2 for m in means) / (len(means) - 1)
    return math.sqrt(var)


def _ss_alpha_helix_count(active_cas):
    """Cheap helix proxy: count residues with an i, i+3 partner at 5.0-5.6 A.
    Active-site residues only."""
    by_chain_seq = sorted([(c["chain"], c["resseq"], i) for i, c in enumerate(active_cas)])
    chain_to_resseqs = defaultdict(dict)
    for chain, rs, idx in by_chain_seq:
        chain_to_resseqs[chain][rs] = idx
    n = 0
    for chain, idxs in chain_to_resseqs.items():
        for rs, i in idxs.items():
            j = idxs.get(rs + 3)
            if j is None: continue
            wi = active_cas[i]["world"]; wj = active_cas[j]["world"]
            d = math.sqrt(sum((wi[k]-wj[k])**2 for k in range(3)))
            if 4.8 <= d <= 5.8:
                n += 1
    return n


def e_dynamics_proxy(atoms, fields, *, return_breakdown: bool = False):
    """Static-sculpture proxy. Reads A_cat.A_dynamics for parameters.
    Returns 0 if no A_dynamics channel set."""
    ad = fields.a.get("channels", {}).get("A_dynamics")
    if not ad:
        return (0.0, {"note": "no A_dynamics channel; term not applicable"}) \
            if return_breakdown else 0.0

    radius = ad.get("active_site_radius_A", 8.0)
    max_var = ad.get("max_bb_neighbor_dist_var_A", 0.6)
    min_ss = ad.get("min_secondary_structure_residues_in_active_site", 4)

    active_cas = _ca_atoms_active_site(atoms, fields, radius_A=radius)
    if len(active_cas) < 5:
        # Too few active-site residues — not enough for a meaningful proxy
        E = 0.0
        if return_breakdown:
            return E, {"n_active_site_residues": len(active_cas),
                       "note": "too few active-site residues for proxy"}
        return E

    var = _bb_neighbor_variance(active_cas, k_neighbors=3)
    helix_count = _ss_alpha_helix_count(active_cas)

    # Per-component penalty (linear above tolerance)
    e_var = max(0.0, (var - max_var) / max(0.1, max_var)) if var is not None else 0.0
    e_ss  = max(0.0, (min_ss - helix_count) / max(1.0, min_ss))

    E = e_var + e_ss
    if return_breakdown:
        return E, {
            "n_active_site_residues": len(active_cas),
            "bb_neighbor_var_A": round(var, 3) if var is not None else None,
            "bb_var_tolerance_A": max_var,
            "E_bb_var": round(e_var, 4),
            "ss_helix_residue_count": helix_count,
            "ss_required": min_ss,
            "E_ss_deficit": round(e_ss, 4),
            "proxy_grade": ad.get("proxy_grade", "cheap"),
        }
    return E


def params_doc():
    return {
        "term": "E_dynamics_proxy",
        "intent": "static-sculpture detector — flags irregular active-site backbones",
        "sign": "higher = irregular geometry = worse",
        "grade": "proxy (cheap, RFD2-output-only)",
        "components": ["BB-neighbor distance variance", "alpha-helix residue count deficit"],
        "honest_limitation": "this is NOT V_preorg. Real dynamics validation requires "
                              "PyRosetta/OpenMM relax + normal modes. This term flags "
                              "the most obvious 'static sculpture' patterns and serves "
                              "in the SMC loop where full V_preorg is too expensive.",
    }
