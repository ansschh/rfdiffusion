#!/usr/bin/env python3
"""e_cat.py - composite catalytic energy.

  E_cat(P; A_cat, S_chem) = sum_t lambda_t * E_t(P, A_cat)

where t in {path, contact, avoid, anchor, seq_chem, site}.

The catalytic likelihood is exp(-E_cat). For SMC / posterior sampling around
RFD2, particles are weighted by exp(-lambda_schedule * E_cat).

LAMBDA SCHEDULE (default tier weights, calibrated from term-by-term probes):
  lambda_path     = 1.0   # path-blocking is universal; scale ~ # heavy in cone
  lambda_contact  = 1.0   # reward correct contacts; oracle scale ~ -4 perfect
  lambda_avoid    = 1.0   # donor poisoning + extra steric
  lambda_anchor   = 1.0   # carried anchor reward; 0 today (no carried anchors)
  lambda_seq_chem = 1.0   # identity-level chemistry penalty
  lambda_site     = 1.0   # coord-geometry violations; categorical

For annealed in-denoiser SMC (Stage 2), this becomes lambda_t schedule:
  early (high noise):  lambda_path = 1.0, all others 0
                        (only the substrate-cone exclusion is meaningful)
  middle:              + lambda_contact, lambda_avoid (pocket topology)
  late (low noise):    + lambda_anchor, lambda_seq_chem, lambda_site
                        (residue-frame chemistry)
"""
from __future__ import annotations
from typing import Iterable, Optional
from guidance.a_cat_fields import ACatFields
from guidance.e_terms.path        import e_path
from guidance.e_terms.contact     import e_contact
from guidance.e_terms.avoid       import e_avoid
from guidance.e_terms.anchor      import e_anchor
from guidance.e_terms.seq_chem    import e_seq_chem
from guidance.e_terms.site        import e_site
from guidance.e_terms.face        import e_face
from guidance.e_terms.coord_zones import e_coord_zones


DEFAULT_LAMBDAS = {
    "path":         1.0,
    "contact":      1.0,
    "avoid":        1.0,
    "anchor":       1.0,
    "seq_chem":     1.0,
    "site":         1.0,
    "face":         1.0,
    "coord_zones":  1.0,
}

# Per-noise-level annealed schedules (the PI's tiered guidance idea).
# COARSE = high-noise: only path + face partition + avoid (cofactor body has shape;
#         protein matter on reactive face is wrong regardless of identity)
# MID = + contact + coord_zones (pocket topology takes shape)
# FINE = + anchor + seq_chem + site (residue-frame chemistry)
SCHEDULE_COARSE = {"path": 1.0, "contact": 0.0, "avoid": 0.5, "anchor": 0.0,
                   "seq_chem": 0.0, "site": 0.0, "face": 1.0, "coord_zones": 0.0}
SCHEDULE_MID    = {"path": 1.0, "contact": 1.0, "avoid": 1.0, "anchor": 0.0,
                   "seq_chem": 0.0, "site": 0.0, "face": 1.0, "coord_zones": 1.0}
SCHEDULE_FINE   = {"path": 1.0, "contact": 1.0, "avoid": 1.0, "anchor": 1.0,
                   "seq_chem": 1.0, "site": 1.0, "face": 1.0, "coord_zones": 1.0}


def e_cat(atoms, fields: ACatFields, *,
          lambdas: Optional[dict] = None,
          return_breakdown: bool = False):
    """E_cat = sum_t lambda_t * E_t. Atoms is iterable of atom dicts; fields is
    an ACatFields. Returns scalar (or scalar + per-term dict if return_breakdown)."""
    L = dict(DEFAULT_LAMBDAS)
    if lambdas:
        L.update(lambdas)
    terms = {}
    if L.get("path",        0) != 0.0: terms["path"]        = e_path(atoms, fields)
    if L.get("contact",     0) != 0.0: terms["contact"]     = e_contact(atoms, fields)
    if L.get("avoid",       0) != 0.0: terms["avoid"]       = e_avoid(atoms, fields)
    if L.get("anchor",      0) != 0.0: terms["anchor"]      = e_anchor(atoms, fields)
    if L.get("seq_chem",    0) != 0.0: terms["seq_chem"]    = e_seq_chem(atoms, fields)
    if L.get("site",        0) != 0.0: terms["site"]        = e_site(atoms, fields)
    if L.get("face",        0) != 0.0: terms["face"]        = e_face(atoms, fields)
    if L.get("coord_zones", 0) != 0.0: terms["coord_zones"] = e_coord_zones(atoms, fields)
    E = sum(L.get(k, 1.0) * v for k, v in terms.items())
    if return_breakdown:
        return E, {"lambdas": L, "terms": {k: round(v, 4) for k, v in terms.items()},
                   "weighted": {k: round(L[k]*v, 4) for k, v in terms.items()},
                   "E_cat": round(E, 4)}
    return E


SCHEDULES = {
    "default": DEFAULT_LAMBDAS,
    "coarse":  SCHEDULE_COARSE,
    "mid":     SCHEDULE_MID,
    "fine":    SCHEDULE_FINE,
}
