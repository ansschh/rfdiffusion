#!/usr/bin/env python3
"""E_site - continuous penalty on cofactor coordination geometry + extra-donor
overshoot. Soft version of V_chem categorical gates.

Sign convention: HIGHER E_site = MORE coordination violations = WORSE.

Components:
  1. metal-identity (hard mismatch -> w_metal large penalty)
  2. coordination number overshoot/undershoot vs template's allowed_cn
  3. per-donor distance deviation outside template's M-X bands
  4. missing required hapticity (eta5_cp absent when required, or no kappa donors)
  5. forbidden hapticity present (e.g., eta5 in a non-Cp template)
  6. missing required active-species elements (e.g., O for aqua site)
  7. EXTRA PROTEIN DONOR penalty - any sidechain donor atom in the coord shell
     that is NOT in the cofactor's curated donor set. This is the
     "designed-residue poisons the active site" signal that RFD2 cannot see
     directly because it only sees the cofactor atoms as fixed.

Loads V_chem rules from pipeline/v_chem_rules.yaml (uses target_to_template).
"""
from __future__ import annotations
import math, os
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

METALS = {"IR","ZN","RH","RU","FE","MN","CU","CO","NI","PD","PT","MO","W","OS","V","CR","MG","CA","NA","K","AL"}
COORD_CUT_HEAVY = 3.0
COORD_CUT_H = 2.0


def _load_rules():
    if not _HAS_YAML:
        raise SystemExit("E_site needs PyYAML.")
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.normpath(os.path.join(here, "..", "..", "v_chem_rules.yaml"))
    with open(p) as f:
        return yaml.safe_load(f)


def _template_for(rules, target):
    pdb_id = (target or "").upper()
    tname = rules["target_to_template"].get(pdb_id)
    if not tname:
        base = pdb_id.split("__")[0]
        tname = rules["target_to_template"].get(base)
    if not tname:
        return None, None
    return tname, rules["templates"].get(tname)


def _find_metal(atoms):
    for a in atoms:
        if a.get("record") == "HETATM" and a.get("resname") != "ORI" and a.get("element") in METALS:
            return a
    # fallback any
    for a in atoms:
        if a.get("element") in METALS:
            return a
    return None


def _dist(a, b):
    return math.sqrt((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2 + (a["z"]-b["z"])**2)


def _coord_sphere(metal, atoms):
    sphere = []
    for a in atoms:
        if a is metal: continue
        if a.get("resname") == "ORI" or a.get("name") == "ORI": continue
        d = _dist(a, metal)
        cut = COORD_CUT_H if a.get("element") == "H" else COORD_CUT_HEAVY
        if d <= cut:
            sphere.append((a, d))
    return sphere


def _detect_eta5(sphere, mc_band, spread_max, min_n=4):
    cs = [(a, d) for a, d in sphere if a.get("element") == "C" and mc_band[0] <= d <= mc_band[1]]
    if len(cs) < min_n: return None
    ds = [d for _, d in cs]
    return {"n": len(cs), "spread": max(ds) - min(ds)} if (max(ds) - min(ds)) <= spread_max else None


def _detect_kappa(sphere, donor_el, n_req):
    ds = [(a, d) for a, d in sphere if a.get("element") == donor_el]
    return ds if len(ds) >= n_req else None


def e_site(atoms, fields, *,
           w_metal: float = 5.0, w_cn: float = 0.5,
           w_dist_out_of_band: float = 1.0, dist_band_slack_A: float = 0.0,
           w_hapticity_missing: float = 2.0, w_hapticity_forbidden: float = 3.0,
           w_active_missing: float = 1.5, w_extra_donor: float = 1.5,
           return_breakdown: bool = False):
    """Score coordination integrity. fields is ACatFields (used for target/frame info)."""
    rules = _load_rules()
    target = fields.target
    tname, template = _template_for(rules, target)
    if template is None:
        E = 0.0
        if return_breakdown:
            return E, {"error": f"no V_chem template for target {target}"}
        return E

    metal = _find_metal(atoms)
    if metal is None:
        if return_breakdown:
            return 100.0, {"error": "no metal atom found"}
        return 100.0
    sphere = _coord_sphere(metal, atoms)

    breakdown = {"template": tname, "metal_observed": metal.get("element"),
                 "cn_observed": len(sphere), "components": {}}

    # 1. metal identity
    exp_el = template["metal"]["element"]
    if metal.get("element") != exp_el:
        e_metal = w_metal
        breakdown["components"]["E_metal"] = {"value": e_metal, "obs": metal.get("element"), "exp": exp_el}
    else:
        e_metal = 0.0
        breakdown["components"]["E_metal"] = {"value": 0.0}

    # 2. CN
    cn = len(sphere)
    allowed = template["metal"].get("allowed_cn", [])
    if allowed:
        if cn < min(allowed):
            e_cn = w_cn * (min(allowed) - cn)
        elif cn > max(allowed):
            e_cn = w_cn * (cn - max(allowed))
        else:
            e_cn = 0.0
    else:
        e_cn = 0.0
    breakdown["components"]["E_cn"] = {"value": e_cn, "cn": cn, "allowed": allowed}

    # 3. M-X distance bands
    bands = template["metal"].get("bond_distances", {})
    e_dist = 0.0
    viols = []
    for a, d in sphere:
        key = f"{exp_el}-{a.get('element')}"
        if key not in bands: continue
        lo, hi = bands[key]
        lo_s, hi_s = lo - dist_band_slack_A, hi + dist_band_slack_A
        if d < lo_s:
            v = lo_s - d
            e_dist += w_dist_out_of_band * v
            viols.append({"atom": a.get("name"), "el": a.get("element"), "d": round(d,3),
                          "band": [lo,hi], "deviation": round(v,3)})
        elif d > hi_s:
            v = d - hi_s
            e_dist += w_dist_out_of_band * v
            viols.append({"atom": a.get("name"), "el": a.get("element"), "d": round(d,3),
                          "band": [lo,hi], "deviation": round(v,3)})
    breakdown["components"]["E_dist"] = {"value": round(e_dist, 4), "violations": viols}

    # 4-5. hapticity
    hap = template.get("hapticity", {})
    e_hap = 0.0
    hap_notes = []
    for req in hap.get("required", []):
        if req["type"] == "eta5_cp":
            r = _detect_eta5(sphere, req["mc_dist_band"], req["mc_spread_max"],
                             min_n=req.get("n_carbons", 5))
            if not r:
                e_hap += w_hapticity_missing
                hap_notes.append(f"missing required eta5_cp")
        elif req["type"] == "kappa_n_n":
            r = _detect_kappa(sphere, req.get("donor_element", "N"), req.get("n_donors", 2))
            if not r:
                e_hap += w_hapticity_missing
                hap_notes.append(f"missing required kappa-{req.get('donor_element', 'N')}")
    for fb in hap.get("forbidden", []):
        if fb["type"] == "eta5_cp":
            r = _detect_eta5(sphere, fb.get("mc_dist_band", [1.95, 2.40]),
                             fb.get("mc_spread_max", 0.30))
            if r:
                e_hap += w_hapticity_forbidden
                hap_notes.append(f"forbidden eta5_cp present")
    breakdown["components"]["E_hapticity"] = {"value": round(e_hap, 4), "notes": hap_notes}

    # 6. active species elements
    active = template.get("active_species", {})
    required_active = active.get("required_elements", [])
    elements_in_sphere = {a.get("element") for a, _ in sphere}
    missing_active = [e for e in required_active if e not in elements_in_sphere]
    e_active = w_active_missing * len(missing_active)
    breakdown["components"]["E_active"] = {"value": e_active, "missing": missing_active}

    # 7. EXTRA PROTEIN DONOR: ATOM record sidechain donor inside coord shell
    BACKBONE = {"N", "CA", "C", "O", "OXT", "H"}
    e_extra = 0.0
    extras = []
    for a, d in sphere:
        if a.get("record") == "ATOM" and a.get("name") not in BACKBONE \
                and a.get("element") in ("N", "O", "S"):
            e_extra += w_extra_donor
            extras.append({"res": f"{a['resname']}{a.get('resseq','')}{a.get('chain','')}",
                            "atom": a.get("name"), "d": round(d, 3)})
    breakdown["components"]["E_extra_donor"] = {"value": round(e_extra, 4), "extras": extras}

    E = e_metal + e_cn + e_dist + e_hap + e_active + e_extra
    breakdown["E_site_total"] = round(E, 4)
    if return_breakdown:
        return E, breakdown
    return E


def params_doc():
    return {
        "term": "E_site",
        "intent": "continuous penalty on coordination geometry violations (soft V_chem)",
        "sign": "higher E_site = more coord violations = worse",
        "weights": {
            "w_metal_wrong": 5.0, "w_cn_per_count": 0.5, "w_dist_per_angstrom_off_band": 1.0,
            "w_hapticity_missing_required": 2.0, "w_hapticity_forbidden_present": 3.0,
            "w_active_missing_element": 1.5, "w_extra_protein_donor_in_coord_shell": 1.5,
        },
        "extra_donor_definition": "ATOM record sidechain N/O/S atom within 3.0 A of metal (heavy cut) or 2.0 A (H cut)",
        "depends_on": "v_chem_rules.yaml (target_to_template, templates)",
    }
