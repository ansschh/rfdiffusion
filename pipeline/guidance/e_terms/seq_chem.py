#!/usr/bin/env python3
"""E_seq_chem - sequence-level chemistry penalty.

Penalizes RESIDUE IDENTITIES (not atom positions) that are chemically
incompatible with the local context, regardless of whether sidechains happen
to point the right way. This is the term LigandMPNN biasing would consume:
at position i, prefer/avoid residue classes based on local cofactor context.

Sign convention: HIGHER E_seq_chem = MORE incompatible identities = WORSE.

v0 rules (DRAFT, metal-class-aware):

Rule 1 - Soft-metal sequence poisoning:
  Soft metals (Ir, Rh, Ru, Pt, Pd) within 6 A of cofactor metal: penalize
    CYS by 1.0, MET by 0.5, HIS by 0.5 (HIS exempted if it sits at carried
    anchor pos and that anchor explicitly allows HIS).
  Hard metals (Zn, Mg, Ca): no soft-donor poisoning, but penalize protonated
    bases (LYS, ARG) within 4 A (they would compete with anchor donors).

Rule 2 - Substrate-cone identity:
  Charged residue (LYS, ARG, ASP, GLU) inside the substrate-approach cone:
  penalize by 0.3 per residue (cationic TS doesn't want charged residues
  in the path).

Rule 3 - Cationic-TS anion stabilization reward (cp_star_ir_iii_ath only):
  ASP/GLU just past cone exit (5-8 A from metal along axis): reward 0.3 per
  residue. Mirrors A_elec channel's "charged_acid expected ~4 A beyond
  hydride" rule but at residue identity level rather than atom level.

E_seq_chem(P) = sum_residues (penalty_per_residue) - sum (rewards)

Natural scale: 1 well-placed Cys -> +1.0; 2 charged in cone -> +0.6;
1 stabilizing ASP past cone -> -0.3.
"""
from __future__ import annotations
import math
from collections import defaultdict

BACKBONE = {"N", "CA", "C", "O", "OXT", "H"}

SOFT_METALS = {"IR", "RH", "RU", "PT", "PD", "OS"}
HARD_METALS = {"ZN", "MG", "CA", "MN", "FE", "CO", "NI", "CU"}   # Cu/Ni borderline

# Cationic-TS reaction classes (matches A_cat instantiator's table)
CATIONIC_TS_TARGETS = {"3ZP9", "5OD5"}


def _residue_pos_local(residue_atoms, fields):
    """Closest-sidechain-heavy in LOCAL coords (or CA for GLY).
    Matches manifest's min_dist_to_core convention so 'residue near metal'
    means 'any sidechain atom is reachable to the metal'."""
    sc = [a for a in residue_atoms if a["element"] != "H" and a["name"] not in BACKBONE]
    if not sc:
        ca = next((a for a in residue_atoms if a["name"] == "CA"), None)
        if ca is None: return None
        return fields.to_local((ca["x"], ca["y"], ca["z"]))
    # find the sidechain atom closest to the metal (origin in local)
    best = (None, 1e18)
    for a in sc:
        pl = fields.to_local((a["x"], a["y"], a["z"]))
        d2 = pl[0]*pl[0] + pl[1]*pl[1] + pl[2]*pl[2]
        if d2 < best[1]:
            best = (pl, d2)
    return best[0]


def _metal_class(fields):
    for c in fields.a.get("cofactor_atoms_local", []):
        el = c.get("element", "").upper()
        if el in SOFT_METALS: return "soft"
        if el in HARD_METALS: return "hard"
    return "soft"  # default conservative


def _is_in_cone(pl, fields):
    """Cone occupancy via the field; > 0.05 counts as 'inside'."""
    # _path is in local; we need world coord for eval_path. Convert back.
    # Easier: just check the cone geometry directly in local frame.
    if not fields._path_ok: return False
    ap = fields._path_apex_geom
    ax = fields._path_axis
    rel = (pl[0]-ap[0], pl[1]-ap[1], pl[2]-ap[2])
    r_par = rel[0]*ax[0] + rel[1]*ax[1] + rel[2]*ax[2]
    if r_par < fields._path_r_min - 0.5 or r_par > fields._path_r_max + 0.5:
        return False
    r_perp = math.sqrt(max(0.0, rel[0]*rel[0]+rel[1]*rel[1]+rel[2]*rel[2] - r_par*r_par))
    cone_r = max(0.0, r_par) * fields._path_tan_half
    return r_perp <= cone_r + 0.5


def _is_just_past_cone(pl, fields, beyond_min=0.0, beyond_max=4.0):
    """Residue centroid in the "stabilization zone" just past the cone exit
    (axial distance r_par in [r_max, r_max + beyond_max])."""
    if not fields._path_ok: return False
    ap = fields._path_apex_geom
    ax = fields._path_axis
    rel = (pl[0]-ap[0], pl[1]-ap[1], pl[2]-ap[2])
    r_par = rel[0]*ax[0] + rel[1]*ax[1] + rel[2]*ax[2]
    if r_par < fields._path_r_max + beyond_min: return False
    if r_par > fields._path_r_max + beyond_max: return False
    r_perp = math.sqrt(max(0.0, rel[0]*rel[0]+rel[1]*rel[1]+rel[2]*rel[2] - r_par*r_par))
    cone_r = max(0.0, r_par) * fields._path_tan_half
    return r_perp <= cone_r + 1.5


def e_seq_chem(atoms, fields, *,
               soft_metal_poison_dist_A: float = 6.0,
               hard_metal_base_dist_A: float = 4.0,
               soft_poison_weights=None,
               hard_base_weight: float = 0.5,
               cone_charge_weight: float = 0.3,
               past_cone_acid_reward: float = 0.3,
               past_cone_base_penalty: float = 0.3,
               aromatic_pi_stack_reward: float = 0.4,
               aromatic_pi_stack_range_A=(3.5, 5.5),
               charge_charge_repulsion_dist_A: float = 5.0,
               charge_charge_repulsion_weight: float = 0.2,
               return_per_residue: bool = False):
    """
    Penalize chemically incompatible residue identities near cofactor.
    """
    if soft_poison_weights is None:
        soft_poison_weights = {"CYS": 1.0, "MET": 0.5, "HIS": 0.5}

    by_res = defaultdict(list)
    for a in atoms:
        if a.get("record") != "ATOM": continue
        by_res[(a["chain"], a["resseq"], a["resname"])].append(a)

    metal_class = _metal_class(fields)
    pdb_id = (fields.target or "").upper().split("__")[0]
    cationic_ts = pdb_id in CATIONIC_TS_TARGETS

    # carried HIS anchor positions (exempt HIS from soft poison if within tolerance)
    his_exempt_local = []
    for an in (fields.A_anchor or []):
        if an.get("carried") and "HIS" in (an.get("allowed_residues") or []):
            if an.get("pos_local"): his_exempt_local.append(tuple(an["pos_local"]))

    E_pen = 0.0
    E_rew = 0.0
    per = [] if return_per_residue else None
    # Pre-compute per-residue position (for charge-charge pass) so we can
    # do the O(N^2) over only charged residues without re-traversing.
    charged_positions = []   # list of (resname, resseq, chain, pos_local, charge_sign)
    for (chain, resseq, resname), ratoms in by_res.items():
        pl = _residue_pos_local(ratoms, fields)
        if pl is None: continue
        d_metal = math.sqrt(pl[0]*pl[0] + pl[1]*pl[1] + pl[2]*pl[2])

        local_pen = 0.0
        local_rew = 0.0
        reasons = []

        # Rule 1: metal-class specific poisoning
        if metal_class == "soft":
            if resname in soft_poison_weights and d_metal <= soft_metal_poison_dist_A:
                exempt = False
                if resname == "HIS" and his_exempt_local:
                    # check if this HIS sits at any carried anchor pos (within 2 A)
                    for ax in his_exempt_local:
                        d_anchor = math.sqrt((pl[0]-ax[0])**2 + (pl[1]-ax[1])**2 + (pl[2]-ax[2])**2)
                        if d_anchor <= 2.0:
                            exempt = True; break
                if not exempt:
                    # decay with distance: full weight at d=metal_atoms, fade by 6 A
                    falloff = max(0.0, 1.0 - (d_metal / soft_metal_poison_dist_A))
                    pen = soft_poison_weights[resname] * falloff
                    local_pen += pen
                    reasons.append(f"soft-metal {resname} poison @{round(d_metal,2)}A: +{round(pen,3)}")
        elif metal_class == "hard":
            if resname in ("LYS", "ARG") and d_metal <= hard_metal_base_dist_A:
                falloff = max(0.0, 1.0 - (d_metal / hard_metal_base_dist_A))
                pen = hard_base_weight * falloff
                local_pen += pen
                reasons.append(f"hard-metal {resname} (base) @{round(d_metal,2)}A: +{round(pen,3)}")

        # Rule 2: charged residue inside substrate cone
        if resname in ("LYS", "ARG", "ASP", "GLU") and _is_in_cone(pl, fields):
            local_pen += cone_charge_weight
            reasons.append(f"charged {resname} in substrate cone: +{cone_charge_weight}")

        # Rule 3: anion stabilization just past cone exit (cationic-TS targets only)
        if cationic_ts and resname in ("ASP", "GLU") and _is_just_past_cone(pl, fields):
            local_rew += past_cone_acid_reward
            reasons.append(f"acid {resname} past-cone (cationic TS stab): -{past_cone_acid_reward}")

        # Rule 4 (NEW): cation destabilizer past cone — LYS/ARG just past cone exit on
        # cationic-TS targets is the opposite of an anion stabilizer. Penalize.
        if cationic_ts and resname in ("LYS", "ARG") and _is_just_past_cone(pl, fields):
            local_pen += past_cone_base_penalty
            reasons.append(f"base {resname} past-cone (cationic TS destab): +{past_cone_base_penalty}")

        # Rule 5 (NEW): pi-stacking aromatic near substrate cone face.
        # PHE/TYR/TRP at 3.5-5.5 A from the cone-exit centerline (perpendicular)
        # = potential pi-stacking with iminium / aryl substrate. Reward.
        if resname in ("PHE", "TYR", "TRP") and fields._path_ok:
            ap = fields._path_apex_geom
            ax = fields._path_axis
            rel = (pl[0]-ap[0], pl[1]-ap[1], pl[2]-ap[2])
            r_par = rel[0]*ax[0] + rel[1]*ax[1] + rel[2]*ax[2]
            if (fields._path_r_min - 0.5) <= r_par <= (fields._path_r_max + 1.0):
                r_perp = math.sqrt(max(0.0, rel[0]*rel[0]+rel[1]*rel[1]+rel[2]*rel[2] - r_par*r_par))
                lo, hi = aromatic_pi_stack_range_A
                if lo <= r_perp <= hi:
                    local_rew += aromatic_pi_stack_reward
                    reasons.append(f"aromatic {resname} pi-stacking range (r_perp={round(r_perp,2)} A): -{aromatic_pi_stack_reward}")

        # Collect charged residues for the O(N^2) charge-charge pass below
        if resname in ("LYS", "ARG"):
            charged_positions.append((resname, resseq, chain, pl, +1))
        elif resname in ("ASP", "GLU"):
            charged_positions.append((resname, resseq, chain, pl, -1))

        if local_pen or local_rew:
            E_pen += local_pen
            E_rew += local_rew
            if per is not None:
                per.append({"chain": chain, "resseq": resseq, "resname": resname,
                            "d_metal": round(d_metal, 3),
                            "penalty": round(local_pen, 4),
                            "reward": round(local_rew, 4),
                            "net": round(local_pen - local_rew, 4),
                            "reasons": reasons})

    # Rule 6 (NEW): charge-charge like-repulsion near active site.
    # Two like-charged residues within charge_charge_repulsion_dist_A of each
    # other and both within 10 A of metal: repulsion penalty (destabilizes
    # the active-site electrostatics).
    cc_pairs = 0
    e_cc = 0.0
    for i in range(len(charged_positions)):
        ri = charged_positions[i]
        d_i_metal = math.sqrt(sum(c*c for c in ri[3]))
        if d_i_metal > 10.0: continue
        for j in range(i+1, len(charged_positions)):
            rj = charged_positions[j]
            if ri[4] != rj[4]:   # opposite sign — attractive, no penalty
                continue
            d_j_metal = math.sqrt(sum(c*c for c in rj[3]))
            if d_j_metal > 10.0: continue
            d_ij = math.sqrt(sum((ri[3][k]-rj[3][k])**2 for k in range(3)))
            if d_ij > charge_charge_repulsion_dist_A: continue
            # Penalty falls off linearly with distance: closer = worse
            pen = charge_charge_repulsion_weight * max(0.0, 1.0 - d_ij / charge_charge_repulsion_dist_A)
            e_cc += pen
            cc_pairs += 1

    E = E_pen + e_cc - E_rew
    if return_per_residue:
        return E, {"E_penalty": round(E_pen, 4),
                   "E_reward": round(E_rew, 4),
                   "E_charge_charge_repulsion": round(e_cc, 4),
                   "n_like_charge_repulsion_pairs": cc_pairs,
                   "metal_class": metal_class, "cationic_ts": cationic_ts,
                   "per_residue": sorted(per, key=lambda x: -abs(x["net"]))[:30]}
    return E


def params_doc():
    return {
        "term": "E_seq_chem",
        "intent": "penalize residue IDENTITIES incompatible with local cofactor chemistry",
        "sign": "higher E_seq_chem = more bad identities = worse",
        "rules": {
            "soft_metal_poison": "CYS 1.0, MET 0.5, HIS 0.5 within 6 A of soft metals (Ir/Rh/Ru/Pt/Pd/Os); HIS exempted if at carried anchor pos",
            "hard_metal_base": "LYS/ARG within 4 A of hard metals (Zn/Mg/Ca/...): 0.5",
            "cone_charge": "charged residue (KRDE) inside substrate cone: 0.3",
            "cationic_ts_stab": "ASP/GLU just past cone exit on cationic-TS targets (3ZP9, 5OD5): reward 0.3",
        },
        "linear_falloff_with_distance": True,
        "extensibility": "Soft/hard metal sets, weights, dist thresholds all parameterizable",
    }
