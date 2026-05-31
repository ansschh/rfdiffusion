#!/usr/bin/env python3
"""E_coord_zones - allowed-zone donors only (PI: enforce allowed donor residues
only in allowed coordination-support zones).

This is a STRICTER refinement of E_avoid's donor poisoning:
  E_avoid penalizes ANY sidechain donor atom within the metal coord shell.
  E_coord_zones penalizes ANY sidechain donor atom OUTSIDE the explicitly
  allowed zones — and exempts donors that ARE inside an allowed zone.

The allowed zones come from A_cat.A_coord_zones, which the instantiator
emits:
  - chem mode: zones at cofactor's own retained N/O donors (the cofactor's
    intrinsic coord sphere is allowed, everything else is poison)
  - any mode: carried-anchor positions (allowed_residues from anchor entry)

Schema for A_coord_zones entries:
  { "center_local": [x,y,z],
    "radius_A": 1.0,
    "allowed_residues": ["HIS", "CYS", ...]    or null (no residue restriction),
    "allowed_donor_atoms": ["NE2", "ND1", ...] or null (any donor atom),
    "source": "cofactor_N_donor" | "carried_anchor" | "user" }

E_coord_zones(P) = sum over protein donor atoms not inside any allowed zone
                   of strength(atom) * 1/(1+d_metal) * gating

Atoms inside an allowed zone (within radius_A of center, with matching
allowed_residues if specified) contribute 0.

Sign: higher = more donors in forbidden positions = worse.

This term REPLACES E_avoid's donor poisoning component when both are active
(set lambda_avoid_donor=0 in e_cat config). Keeping E_avoid for steric clash
+ E_coord_zones for donor placement is the cleanest separation.
"""
from __future__ import annotations
import math

# donor strength table — same as E_avoid for consistency
DONOR_STRENGTH = {
    ("HIS", "ND1"): 1.5, ("HIS", "NE2"): 1.5,
    ("ASP", "OD1"): 1.0, ("ASP", "OD2"): 1.0,
    ("GLU", "OE1"): 1.0, ("GLU", "OE2"): 1.0,
    ("SER", "OG"):  0.5,
    ("THR", "OG1"): 0.5,
    ("TYR", "OH"):  0.7,
    ("ASN", "OD1"): 0.4, ("ASN", "ND2"): 0.3,
    ("GLN", "OE1"): 0.4, ("GLN", "NE2"): 0.3,
    ("CYS", "SG"):  2.0,
    ("MET", "SD"):  1.2,
    ("LYS", "NZ"):  0.8,
    ("ARG", "NH1"): 0.5, ("ARG", "NH2"): 0.5, ("ARG", "NE"): 0.3,
    ("TRP", "NE1"): 0.4,
}


def e_coord_zones(atoms, fields, *,
                  max_donor_check_dist_A: float = 5.0,
                  zone_default_radius_A: float = 1.0,
                  return_per_donor: bool = False):
    """For each protein sidechain donor atom within max_donor_check_dist_A of
    the metal: contribute penalty unless it's inside an allowed zone.

    Reads zones from fields.a['channels'].get('A_coord_zones', []) — emitted
    by the extended instantiate_acat.
    """
    zones = fields.a.get("channels", {}).get("A_coord_zones") or []
    if not zones:
        # no zones defined: term is a no-op (different from E_avoid which has
        # a generic shell penalty)
        return (0.0, {"n_donors_checked": 0, "n_inside_zone": 0,
                      "n_outside_zone": 0, "per_donor": []}) if return_per_donor else 0.0

    total = 0.0
    per_donor = []
    n_checked = 0; n_inside = 0; n_outside = 0
    for a in atoms:
        if a.get("record") != "ATOM": continue
        nm = a.get("name"); resname = a.get("resname")
        key = (resname, nm)
        if key not in DONOR_STRENGTH: continue
        strength = DONOR_STRENGTH[key]
        p = (a["x"], a["y"], a["z"])
        pl = fields.to_local(p)
        d_metal = math.sqrt(pl[0]*pl[0] + pl[1]*pl[1] + pl[2]*pl[2])
        if d_metal > max_donor_check_dist_A: continue
        n_checked += 1

        # check each allowed zone
        inside_any = False
        for z in zones:
            c = z.get("center_local")
            if c is None: continue
            r = z.get("radius_A", zone_default_radius_A)
            dx = pl[0]-c[0]; dy = pl[1]-c[1]; dz = pl[2]-c[2]
            d_zone = math.sqrt(dx*dx + dy*dy + dz*dz)
            if d_zone > r: continue
            allowed_res = z.get("allowed_residues")
            if allowed_res and resname not in allowed_res: continue
            allowed_atoms = z.get("allowed_donor_atoms")
            if allowed_atoms and nm not in allowed_atoms: continue
            inside_any = True
            break

        if inside_any:
            n_inside += 1
            if return_per_donor:
                per_donor.append({"res": f"{resname}{a.get('resseq','')}{a.get('chain','')}",
                                   "atom": nm, "d_metal": round(d_metal, 3),
                                   "status": "allowed_zone", "contrib": 0.0})
        else:
            n_outside += 1
            pen = strength / (1.0 + d_metal)   # at 2 A: 1/3; at 5 A: 1/6
            total += pen
            if return_per_donor:
                per_donor.append({"res": f"{resname}{a.get('resseq','')}{a.get('chain','')}",
                                   "atom": nm, "d_metal": round(d_metal, 3),
                                   "donor_strength": strength,
                                   "status": "outside_all_zones",
                                   "contrib": round(pen, 4)})

    if return_per_donor:
        return total, {"n_donors_checked": n_checked,
                       "n_inside_zone": n_inside,
                       "n_outside_zone": n_outside,
                       "per_donor": per_donor}
    return total


def params_doc():
    return {
        "term": "E_coord_zones",
        "intent": "donors only allowed in explicit coord-support zones",
        "sign": "higher = more donors outside allowed zones = worse",
        "active_when": "A_cat.A_coord_zones populated; else no-op",
        "scoring": "sum over protein sidechain donors NOT in allowed zone of strength/(1+d_metal)",
        "exemption": "atoms inside an allowed zone (matching allowed_residues + allowed_donor_atoms if specified) contribute 0",
        "donor_strengths": "same table as E_avoid (CYS 2.0, MET 1.2, HIS 1.5, ASP/GLU 1.0, ...)",
        "max_check_dist_A": 5.0,
    }
