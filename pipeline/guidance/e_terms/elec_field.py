#!/usr/bin/env python3
"""E_elec_field - electrostatic preorganization for cationic-TS reactions.

PI flagged this as the next missing channel after the chem-mode Stage 2c
showed E_face alone doesn't discriminate real vs flipped under chem-mode A_cat
(missing the contact-pull that creates the contrast in oracle).

Model (v1, point charge + protein-interior dielectric):

  For each protein residue r with charge q_r:
      vec(r) = TS_center - residue_sidechain_centroid
      |r|    = ||vec(r)||
      E_field_from_r = q_r * vec(r)/|vec(r)|^3 / epsilon   (Coulomb, atomic units cancelled out)

  Net field at TS center:
      E_net = sum_r E_field_from_r

  Field component along the TS dipole:
      E_along_dipole = E_net . dipole_direction

  Stabilization energy = -|dipole| * E_along_dipole
      (a field aligned with the dipole stabilizes; opposite destabilizes)

  E_elec_field = -stabilization = +|dipole| * E_along_dipole
      Sign convention: HIGHER E_elec_field = MORE destabilization = WORSE.

Residue charge model (v1, naive):
  LYS  +1, ARG  +1                        (protonated bases)
  ASP  -1, GLU  -1                        (deprotonated acids)
  HIS   0 (could be +1 at low pH; v1 neutral; future: PROPKA)
  All others: 0

The TS dipole direction + magnitude + position come from A_cat.A_elec_field,
emitted by instantiate_acat for cationic-TS target classes.
"""
from __future__ import annotations
import math
from collections import defaultdict

BACKBONE = {"N", "CA", "C", "O", "OXT", "H"}

# Per-residue formal charge (v1; PROPKA refinement is future work)
RESIDUE_CHARGE = {
    "LYS":  +1.0, "ARG":  +1.0,
    "ASP":  -1.0, "GLU":  -1.0,
    "HIS":   0.0,             # neutral imidazole; conditional +1 deferred
}


def _sidechain_centroid_world(residue_atoms):
    sc = [a for a in residue_atoms if a["element"] != "H" and a["name"] not in BACKBONE]
    if not sc:
        ca = next((a for a in residue_atoms if a["name"] == "CA"), None)
        if ca is None: return None
        return (ca["x"], ca["y"], ca["z"])
    n = len(sc)
    return (sum(a["x"] for a in sc)/n,
            sum(a["y"] for a in sc)/n,
            sum(a["z"] for a in sc)/n)


def e_elec_field(atoms, fields, *,
                 max_residue_distance_A: float = 15.0,
                 return_per_residue: bool = False):
    """Coulomb-stabilization energy of a TS dipole under the protein's
    distribution of formal charges.

    Sign: higher E_elec_field = more destabilization = worse.
    Lower (negative) = more stabilization = better.

    Reads A_cat.A_elec_field (emitted by instantiate_acat for cationic-TS
    targets). Returns 0 if no A_elec_field present.
    """
    ef = fields.a.get("channels", {}).get("A_elec_field")
    if not ef:
        return (0.0, {"note": "no A_elec_field channel; term not applicable"}) \
            if return_per_residue else 0.0

    ts_center_local = ef.get("ts_center_local", [0.0, 0.0, 5.5])
    dipole_dir_local = ef.get("ts_dipole_direction_local", [0.0, 0.0, 1.0])
    dipole_mag = ef.get("ts_dipole_magnitude_D", 5.0)
    epsilon = ef.get("epsilon_dielectric", 4.0)
    max_d = ef.get("max_residue_distance_A", max_residue_distance_A)

    # Transform TS center + dipole direction from local frame to world coords.
    # ACatFields stores frame.R (rows = local axes in world), frame.origin_world.
    R = fields.R           # rows of R = local axes in world
    o = fields.origin
    # world point = R^T @ local + origin  (R is orthonormal; R^T is its inverse)
    ts_world = (
        R[0][0]*ts_center_local[0] + R[1][0]*ts_center_local[1] + R[2][0]*ts_center_local[2] + o[0],
        R[0][1]*ts_center_local[0] + R[1][1]*ts_center_local[1] + R[2][1]*ts_center_local[2] + o[1],
        R[0][2]*ts_center_local[0] + R[1][2]*ts_center_local[1] + R[2][2]*ts_center_local[2] + o[2],
    )
    dipole_world = (
        R[0][0]*dipole_dir_local[0] + R[1][0]*dipole_dir_local[1] + R[2][0]*dipole_dir_local[2],
        R[0][1]*dipole_dir_local[0] + R[1][1]*dipole_dir_local[1] + R[2][1]*dipole_dir_local[2],
        R[0][2]*dipole_dir_local[0] + R[1][2]*dipole_dir_local[1] + R[2][2]*dipole_dir_local[2],
    )
    # Normalize dipole direction
    dn = math.sqrt(sum(c*c for c in dipole_world))
    if dn > 0:
        dipole_world = tuple(c/dn for c in dipole_world)
    else:
        dipole_world = (0.0, 0.0, 1.0)

    # Aggregate residues
    by_res = defaultdict(list)
    for a in atoms:
        if a.get("record") != "ATOM": continue
        by_res[(a["chain"], a["resseq"], a["resname"])].append(a)

    e_along = 0.0   # net field component along dipole direction
    per_res = []
    for (chain, resseq, resname), ratoms in by_res.items():
        q = RESIDUE_CHARGE.get(resname, 0.0)
        if q == 0.0:
            continue
        cw = _sidechain_centroid_world(ratoms)
        if cw is None: continue
        rx, ry, rz = ts_world[0]-cw[0], ts_world[1]-cw[1], ts_world[2]-cw[2]
        r = math.sqrt(rx*rx + ry*ry + rz*rz)
        if r > max_d or r < 0.5:   # too far or unphysically close
            continue
        # Coulomb field vector at TS from a unit charge at cw:
        #    E_field = q * (TS - cw) / |r|^3 / epsilon
        # we want the component along the dipole direction
        e_field_along = (q / (epsilon * r * r * r)) * \
                        (rx * dipole_world[0] + ry * dipole_world[1] + rz * dipole_world[2])
        e_along += e_field_along
        if return_per_residue:
            per_res.append({
                "residue": f"{resname}{resseq}{chain}",
                "charge": q, "d_TS": round(r, 3),
                "contrib_e_along": round(e_field_along, 5),
            })

    # Stabilization = -dipole_magnitude * e_field_along_dipole
    # E_elec_field = -stabilization (so higher = worse)
    E = dipole_mag * e_along

    if return_per_residue:
        return E, {
            "E_elec_field": round(E, 4),
            "dipole_magnitude_D": dipole_mag,
            "ts_world": ts_world,
            "dipole_world": dipole_world,
            "epsilon": epsilon,
            "net_field_along_dipole": round(e_along, 5),
            "n_charged_residues": len(per_res),
            "top_destabilizers": sorted([r for r in per_res if r["contrib_e_along"] > 0],
                                         key=lambda r: -r["contrib_e_along"])[:5],
            "top_stabilizers": sorted([r for r in per_res if r["contrib_e_along"] < 0],
                                       key=lambda r: r["contrib_e_along"])[:5],
        }
    return E


def params_doc():
    return {
        "term": "E_elec_field",
        "intent": "electrostatic preorganization for cationic-TS stabilization",
        "sign": "higher = destabilization = worse",
        "model": "point-charge protein residues; Coulomb 1/r^2 field at TS center; epsilon=4",
        "active_when": "A_cat.A_elec_field populated (cationic-TS target classes only)",
        "charge_model": "LYS/ARG +1, ASP/GLU -1, HIS 0 (neutral imidazole), other 0",
        "max_residue_distance_A": 15.0,
        "v1_simplifications": [
            "no pKa shifts (PROPKA refinement deferred)",
            "no protein-solvent dielectric boundary",
            "no induced dipole or polarizability",
            "no specific H-bond geometry to TS",
            "point-charge model; partial atomic charges not used",
        ],
    }
