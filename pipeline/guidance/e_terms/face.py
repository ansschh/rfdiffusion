#!/usr/bin/env python3
"""E_face - proximal/distal asymmetry. The load-bearing channel PI flagged
after the 2CCY heme transplant failure.

Two halves of a single asymmetry concept:

  REACTIVE FACE (distal side, +z half-space along the A_face_axis):
    where substrate approaches. Protein matter here BLOCKS chemistry.
    Penalize sidechain centroids in this hemisphere proportional to 1/(1+r).
    Distinct from E_path (which penalizes ONLY inside the strict cone):
    E_face_reactive captures the broader "don't crowd the reactive
    hemisphere" signal — including residues beside but not inside the cone
    that would still occlude lateral substrate access.

  PACKING FACE (proximal side, -z half-space):
    where the cofactor body (Cp* ring, axial ligand, etc.) sits. The host
    protein is EXPECTED to bury the cofactor here. Hydrophobic + aromatic
    sidechain mass in this hemisphere is REWARDED, missing density penalized.

E_face(P) = w_reactive * sum(SC_density on reactive face / (1+r))
          - w_packing  * sum(hydrophobic_SC_density on packing face within R_burial)

Sign: higher E_face = more reactive-face occupation OR less packing-face
burial = worse.

THE 2CCY HEME-TRANSPLANT FAILURE this captures:
  Heme's proximal His sits at heme's -z (proximal). When the retrieval
  rotation R mapped heme's proximal His into the 3ZP9 query frame, it
  landed at 3ZP9's +z — the REACTIVE face. So a heme-derived motif places
  a His exactly where 3ZP9 needs an open substrate path. E_face fires hard:
  reactive face has a sidechain at coord-shell distance (penalty ~1.5),
  AND the corresponding packing face is empty (no reward). Net E_face > 1.
  V_path alone missed this because the His sits on the cone EDGE, not
  inside it; the broader hemisphere penalty catches it.
"""
from __future__ import annotations
import math
from collections import defaultdict

BACKBONE = {"N", "CA", "C", "O", "OXT", "H"}

# residue types from RESIDUE_TYPE (matches contact.py / instantiate_acat)
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PRO"}
AROMATIC    = {"PHE", "TYR", "TRP"}
PACKING_RESIDUES = HYDROPHOBIC | AROMATIC


def _sidechain_centroid_local(residue_atoms, fields):
    sc = [a for a in residue_atoms if a["element"] != "H" and a["name"] not in BACKBONE]
    if not sc:
        ca = next((a for a in residue_atoms if a["name"] == "CA"), None)
        if ca is None: return None
        return fields.to_local((ca["x"], ca["y"], ca["z"]))
    n = len(sc)
    cw = (sum(a["x"] for a in sc)/n, sum(a["y"] for a in sc)/n, sum(a["z"] for a in sc)/n)
    return fields.to_local(cw)


def e_face(atoms, fields, *,
           w_reactive: float = 1.0,
           w_packing: float = 0.5,
           reactive_max_dist_A: float = 9.0,
           packing_burial_radius_A: float = 7.0,
           cone_exclusion_zone: bool = True,
           return_per_residue: bool = False):
    """
    Reactive face = +z half-space in local frame (along A_face_axis, which
    defaults to A_path's axis_local, which is metal -> open-site).
    Packing face = -z half-space.

    Args:
      reactive_max_dist_A:  only count reactive-face residues within this distance
                            from metal (further out is solvent, not blocking)
      packing_burial_radius_A: only count packing-face residues within this radius
                            from metal as burial mass
      cone_exclusion_zone:  if True, residues inside the substrate cone are excluded
                            from E_face_reactive (E_path already penalizes them; avoid
                            double-counting).
    """
    # face axis in local frame: prefer A_face channel (the explicit proximal/distal
    # axis set by the instantiator and perturbed by damage_flip_face). Fall back to
    # A_path's axis_local if A_face is absent; finally fall back to +z.
    a_face = fields.a.get("channels", {}).get("A_face") or {}
    if "reactive_axis_local" in a_face:
        axis = tuple(a_face["reactive_axis_local"])
        # normalize
        an = math.sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])
        if an > 0: axis = (axis[0]/an, axis[1]/an, axis[2]/an)
        else:      axis = (0.0, 0.0, 1.0)
    elif fields._path_ok:
        axis = fields._path_axis
    else:
        axis = (0.0, 0.0, 1.0)

    by_res = defaultdict(list)
    for a in atoms:
        if a.get("record") != "ATOM": continue
        by_res[(a["chain"], a["resseq"], a["resname"])].append(a)

    e_react = 0.0
    e_pack  = 0.0
    per = []
    for (chain, resseq, resname), ratoms in by_res.items():
        pl = _sidechain_centroid_local(ratoms, fields)
        if pl is None: continue
        rpar = pl[0]*axis[0] + pl[1]*axis[1] + pl[2]*axis[2]   # projection along face axis
        rperp2 = pl[0]*pl[0] + pl[1]*pl[1] + pl[2]*pl[2] - rpar*rpar
        rperp = math.sqrt(max(0.0, rperp2))
        d_metal = math.sqrt(pl[0]*pl[0] + pl[1]*pl[1] + pl[2]*pl[2])

        # reactive face: rpar > 0 (along the face axis, on the open side)
        if rpar > 0 and d_metal <= reactive_max_dist_A:
            # optionally exclude residues already inside the substrate cone (E_path handles them)
            in_cone = False
            if cone_exclusion_zone and fields._path_ok:
                if (rpar >= fields._path_r_min and rpar <= fields._path_r_max
                        and rperp <= rpar * fields._path_tan_half):
                    in_cone = True
            if not in_cone:
                # penalty falls off as 1 / (1 + r_metal); residues at 2 A from metal score 1/3,
                # residues at 5 A score 1/6
                pen = w_reactive / (1.0 + d_metal)
                e_react += pen
                if return_per_residue:
                    per.append({"chain": chain, "resseq": resseq, "resname": resname,
                                "face": "reactive", "d_metal": round(d_metal, 3),
                                "rpar": round(rpar, 3), "contrib": round(pen, 4)})

        # packing face: rpar < 0 (opposite side from substrate)
        elif rpar < 0 and d_metal <= packing_burial_radius_A and resname in PACKING_RESIDUES:
            # reward for hydrophobic + aromatic burial on the packing face
            # uniform reward per packing residue within burial radius
            rew = w_packing
            e_pack += rew
            if return_per_residue:
                per.append({"chain": chain, "resseq": resseq, "resname": resname,
                            "face": "packing", "d_metal": round(d_metal, 3),
                            "rpar": round(rpar, 3), "contrib": -round(rew, 4)})

    E = e_react - e_pack
    if return_per_residue:
        return E, {"E_reactive_face": round(e_react, 4),
                   "E_packing_face_reward": round(e_pack, 4),
                   "n_reactive_hits": sum(1 for x in per if x["face"] == "reactive"),
                   "n_packing_hits":  sum(1 for x in per if x["face"] == "packing"),
                   "per_residue": sorted(per, key=lambda x: -abs(x["contrib"]))[:25]}
    return E


def params_doc():
    return {
        "term": "E_face",
        "intent": "proximal/distal asymmetry — penalize protein on reactive face, reward packing on opposite face",
        "sign": "higher E_face = reactive face crowded or packing face empty = worse",
        "captures": "the 2CCY heme-transplant failure: heme proximal residues mapped to 3ZP9's reactive face; E_path missed because they sit on cone edge, but the broader hemisphere penalty catches it",
        "face_axis": "ACatFields A_path axis_local; defaults to +z if no A_path",
        "weights": {"w_reactive": 1.0, "w_packing": 0.5,
                    "reactive_max_dist_A": 9.0, "packing_burial_radius_A": 7.0},
        "cone_exclusion": "residues inside the substrate cone are excluded (E_path handles them; avoid double-count)",
        "packing_residues": ["ALA","VAL","LEU","ILE","MET","PRO","PHE","TYR","TRP"],
    }
