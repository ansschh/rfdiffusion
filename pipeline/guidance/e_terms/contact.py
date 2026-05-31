#!/usr/bin/env python3
"""E_contact - reward correct typed residue placement near cofactor.

Sign convention: HIGHER E_contact = FEWER good contacts = WORSE. The energy is
the *negative* of the reward, so the catalytic likelihood becomes higher when
the protein presents the right kind of residue near each A_contact Gaussian.

Per-residue, per-Gaussian:
  reward(residue, Gaussian) =
      w(Gaussian) * exp(-0.5 * |centroid_local - mu_local|^2 / sigma^2)
  where the residue's type must match the Gaussian's type.

Aggregation across residues (per Gaussian): WINNER-TAKE-ALL (max over residues)
  - matches the retrieval scorer (avoids pocket-size bias)
  - alternative `sum` mode sums all matching residues (set agg='sum')

Aggregation across Gaussians: SUM (each Gaussian asks for one good contact).

E_contact(P) = - sum_g max_r reward(r, g)

So if all 4 oracle Gaussians find a perfectly-matching residue at mu, the
reward sum is ~ sum(w_g) ~ 4.0 and E_contact ~ -4.0.

Residue type classification (matches instantiate_acat.RESIDUE_TYPE):
"""
from __future__ import annotations
import math
from collections import defaultdict

RESIDUE_TYPE = {
    "ALA": "hydrophobic", "VAL": "hydrophobic", "LEU": "hydrophobic",
    "ILE": "hydrophobic", "MET": "hydrophobic", "PRO": "hydrophobic",
    "GLY": "small",
    "PHE": "aromatic", "TYR": "aromatic", "TRP": "aromatic",
    "SER": "polar", "THR": "polar", "ASN": "polar", "GLN": "polar", "CYS": "polar",
    "HIS": "anchor",
    "LYS": "charged_base", "ARG": "charged_base",
    "ASP": "charged_acid", "GLU": "charged_acid",
}

BACKBONE = {"N", "CA", "C", "O", "OXT", "H"}


def _sidechain_centroid_world(residue_atoms):
    sc = [a for a in residue_atoms
          if a["element"] != "H" and a["name"] not in BACKBONE]
    if not sc:   # GLY: use CA
        ca = next((a for a in residue_atoms if a["name"] == "CA"), None)
        if ca is None: return None
        return (ca["x"], ca["y"], ca["z"])
    n = len(sc)
    return (sum(a["x"] for a in sc)/n,
            sum(a["y"] for a in sc)/n,
            sum(a["z"] for a in sc)/n)


def _gaussian_at(centroid_local, mu_local, sigma_diag, sigma_floor=0.5):
    sx = max(sigma_floor, sigma_diag[0])
    sy = max(sigma_floor, sigma_diag[1])
    sz = max(sigma_floor, sigma_diag[2])
    dx = (centroid_local[0]-mu_local[0])/sx
    dy = (centroid_local[1]-mu_local[1])/sy
    dz = (centroid_local[2]-mu_local[2])/sz
    return math.exp(-0.5 * (dx*dx + dy*dy + dz*dz))


def e_contact(atoms, fields, *, agg: str = "winner_take_all",
              sigma_floor: float = 0.5,
              return_per_gaussian: bool = False):
    """
    atoms: iterable of atom dicts (protein only - HETATM/LIG ignored).
    fields: ACatFields.
    agg:   'winner_take_all' (default) or 'sum'.

    Returns float E_contact (negative reward) or (E, per_gaussian_list).
    """
    if not fields.A_contact:
        return (0.0, []) if return_per_gaussian else 0.0

    by_res = defaultdict(list)
    for a in atoms:
        if a.get("record") != "ATOM": continue
        by_res[(a["chain"], a["resseq"], a["resname"])].append(a)

    residue_typed = []
    for (chain, resseq, resname), ratoms in by_res.items():
        t = RESIDUE_TYPE.get(resname)
        if t is None: continue
        cw = _sidechain_centroid_world(ratoms)
        if cw is None: continue
        cl = fields.to_local(cw)
        residue_typed.append({"chain": chain, "resseq": resseq, "resname": resname,
                              "type": t, "centroid_local": cl})

    per = []
    total_reward = 0.0
    for g in fields.A_contact:
        gtype = g.get("type")
        rewards = []
        for r in residue_typed:
            if r["type"] != gtype: continue
            reward = g.get("w", 1.0) * _gaussian_at(
                r["centroid_local"], g["mu_local"], g["Sigma_diag"], sigma_floor)
            if reward > 1e-4:
                rewards.append((reward, r))
        if not rewards:
            best_r, best_v = None, 0.0
        else:
            rewards.sort(key=lambda x: -x[0])
            if agg == "winner_take_all":
                best_v = rewards[0][0]; best_r = rewards[0][1]
            elif agg == "sum":
                best_v = sum(x[0] for x in rewards); best_r = rewards[0][1]
            else:
                raise ValueError(f"unknown agg: {agg}")
        total_reward += best_v
        if return_per_gaussian:
            per.append({"type": gtype, "mu_local": g["mu_local"],
                        "best_reward": round(best_v, 4),
                        "best_residue": (f"{best_r['resname']}{best_r['resseq']}{best_r['chain']}"
                                          if best_r else None),
                        "source": g.get("source_residue") or g.get("source_rule", "")})
    E = -total_reward
    if return_per_gaussian:
        return E, per
    return E


def params_doc():
    return {
        "term": "E_contact",
        "intent": "reward correct typed-residue placement near cofactor (NEG energy = REWARD)",
        "sign": "lower E_contact (more negative) = more good contacts = better",
        "field_used": "ACatFields.A_contact (typed Gaussians from instantiate_acat oracle or chem mode)",
        "scoring_unit": "per protein residue, sidechain centroid (GLY -> CA)",
        "agg_across_residues_per_gaussian": "winner_take_all (max); 'sum' option available",
        "agg_across_gaussians": "sum",
        "residue_types_recognized": list(set(RESIDUE_TYPE.values())),
        "sigma_floor_A": 0.5,
        "natural_scale": "perfect match on all oracle Gaussians (4 typical) -> E_contact ~ -4",
    }
