#!/usr/bin/env python3
"""V_chem v0 — gated grammar for organometallic ArM cofactor validity (Rev2 crit-1/2/active).

Categorical PASS/FAIL per gate (G_metal, G_coord, G_hapticity, G_active_state). G_access
lives in v_rxn.py. Wrong-metal must trip a gate, not just get a worse continuous score.

Usage:
  python v_chem.py <design_dir>                              # score one target, write scores_vchem.json
  python v_chem.py --compare <dir1> <dir2> ...               # tabulate gate-pass fractions

Looks for RFD2 outputs (*-atomized-bb-False.pdb); falls back to motif.pdb so the rigid
motif itself can be sanity-checked locally (same metal/cofactor as RFD2 designs).
"""
from __future__ import annotations
import argparse, glob, json, math, os
try:
    import yaml
except ImportError:
    raise SystemExit("V_chem needs PyYAML — pip install pyyaml (boltz-venv has it).")

METALS = {"IR","ZN","RH","RU","FE","MN","CU","CO","NI","PD","PT","MO","W","OS","V","CR","MG","CA","NA","K","AL"}
DETECT_CUT_HEAVY = 3.0
DETECT_CUT_H = 2.0


def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def parse_design(path):
    out = []
    for line in open(path):
        r = line[:6].strip()
        if r not in ("ATOM", "HETATM"):
            continue
        try:
            name = line[12:16].strip()
            el = line[76:78].strip().upper() or "".join(c for c in name if c.isalpha())[:2].upper()
            out.append((r, name, el, line[17:20].strip(),
                        float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            continue
    return out


def find_metal(atoms):
    cands = [a for a in atoms if a[0] == "HETATM" and a[3] not in ("ORI",) and a[2] in METALS]
    if not cands:
        cands = [a for a in atoms if a[2] in METALS]
    return cands[0] if cands else None


def coord_sphere(metal, atoms):
    m = (metal[4], metal[5], metal[6])
    out = []
    for a in atoms:
        if a is metal or a[3] == "ORI" or a[1] == "ORI":   # skip the centering token (not a donor)
            continue
        d = dist(m, (a[4], a[5], a[6]))
        cut = DETECT_CUT_H if a[2] == "H" else DETECT_CUT_HEAVY
        if d <= cut:
            out.append((a, d))
    return out


def detect_eta5(sphere, mc_band, spread_max, min_n=4):
    cs = [(a, d) for a, d in sphere if a[2] == "C" and mc_band[0] <= d <= mc_band[1]]
    if len(cs) < min_n:
        return None
    ds = [d for _, d in cs]
    spread = max(ds) - min(ds)
    if spread > spread_max:
        return None
    return {"n_carbons": len(cs), "mc_min": round(min(ds), 3),
            "mc_max": round(max(ds), 3), "spread": round(spread, 3)}


def detect_kappa_donors(sphere, donor_element, n_required):
    ds = [(a, d) for a, d in sphere if a[2] == donor_element]
    return ds if len(ds) >= n_required else None


def eval_gates(metal, sphere, template):
    res = {}
    obs_el = metal[2]
    exp_el = template["metal"]["element"]

    # G_metal
    res["G_metal"] = ({"pass": True} if obs_el == exp_el
                     else {"pass": False, "reason": f"observed {obs_el} != expected {exp_el}"})

    # G_coord
    cn = len(sphere)
    allowed_cn = template["metal"].get("allowed_cn", [])
    cn_ok = (not allowed_cn) or (cn in allowed_cn)
    bands = template["metal"].get("bond_distances", {})
    dist_viol = []
    for a, d in sphere:
        key = f"{exp_el}-{a[2]}"
        if key in bands and not (bands[key][0] <= d <= bands[key][1]):
            dist_viol.append(f"{a[2]} at {round(d,2)} outside {key}{bands[key]}")
    if cn_ok and not dist_viol:
        res["G_coord"] = {"pass": True, "cn": cn}
    else:
        reasons = ([f"CN={cn} not in {allowed_cn}"] if not cn_ok else []) + dist_viol
        res["G_coord"] = {"pass": False, "cn": cn, "reason": " | ".join(reasons)}

    # G_hapticity (required present + forbidden absent + forbidden_metals rule)
    findings, fails = [], []
    for req in template.get("hapticity", {}).get("required", []):
        if req["type"] == "eta5_cp":
            e5 = detect_eta5(sphere, req["mc_dist_band"], req["mc_spread_max"],
                             min_n=req.get("n_carbons", 5))
            if e5:
                findings.append({"eta5_cp": e5})
                if obs_el in template.get("forbidden_metals", []):
                    fails.append(f"eta5_cp present but observed metal {obs_el} cannot form organometallic eta5")
            else:
                fails.append(f"required eta5_cp not detected (need >={req.get('n_carbons',5)} C in {req['mc_dist_band']} A, spread <= {req['mc_spread_max']})")
        elif req["type"] == "kappa_n_n":
            need = req.get("n_donors", 2)
            de = req.get("donor_element", "N")
            kd = detect_kappa_donors(sphere, de, need)
            if kd:
                findings.append({"kappa_donors": [{"el": a[2], "name": a[1], "d": round(d,3)} for a, d in kd]})
            else:
                fails.append(f"required kappa-{de},{de}: only {sum(1 for a,_ in sphere if a[2]==de)} {de} donors (need {need})")
    for fb in template.get("hapticity", {}).get("forbidden", []):
        if fb["type"] == "eta5_cp":
            e5 = detect_eta5(sphere, fb.get("mc_dist_band", [1.95, 2.40]),
                             fb.get("mc_spread_max", 0.30))
            if e5:
                fails.append(f"forbidden eta5_cp present: {e5}")
    res["G_hapticity"] = ({"pass": True, "findings": findings} if not fails
                         else {"pass": False, "findings": findings, "reason": "; ".join(fails)})

    # G_active_state — required elements present in coord sphere
    active = template.get("active_species", {})
    required = active.get("required_elements", [])
    found = {a[2] for a, _ in sphere}
    missing = [e for e in required if e not in found]
    res["G_active_state"] = ({"pass": True} if not missing
                            else {"pass": False, "reason": f"required active-species element(s) absent from coord sphere: {missing}"})

    res["all_pass"] = all(g["pass"] for g in (res["G_metal"], res["G_coord"],
                                              res["G_hapticity"], res["G_active_state"]))
    return res


def score_design(path, template):
    atoms = parse_design(path)
    metal = find_metal(atoms)
    if not metal:
        return {"file": os.path.basename(path), "error": "no metal found", "all_pass": False}
    sphere = coord_sphere(metal, atoms)
    gates = eval_gates(metal, sphere, template)
    return {
        "file": os.path.basename(path),
        "metal": metal[2],
        "cn_observed": len(sphere),
        "donors": [{"el": a[2], "name": a[1], "d": round(d, 3)} for a, d in sphere],
        "gates": gates,
        "all_pass": gates["all_pass"],
    }


def summarize(target, scores, template_name):
    n = len(scores)
    gate_names = ["G_metal", "G_coord", "G_hapticity", "G_active_state"]
    if n == 0:
        return {"target": target, "template": template_name, "n_designs": 0}
    counts = {g: sum(1 for s in scores if s.get("gates", {}).get(g, {}).get("pass")) for g in gate_names}
    all_pass = sum(1 for s in scores if s.get("all_pass"))
    return {
        "target": target,
        "template": template_name,
        "n_designs": n,
        "frac_all_pass": round(all_pass / n, 3),
        "frac_per_gate": {g: round(counts[g] / n, 3) for g in gate_names},
        "example_failure_reason": {
            g: next((s["gates"].get(g, {}).get("reason", "")
                     for s in scores if not s.get("gates", {}).get(g, {}).get("pass")), None)
            for g in gate_names
        },
    }


def resolve_motif_pdb(d):
    """Find canonical motif.pdb for a target. Accepts either a compiled-motif dir or a
    design dir. V_chem ALWAYS scores the motif (not RFD2 designs) because the cofactor is
    rigid and identical across all 100 designs of a target; and because RFD2 strips H atoms
    from its output so the design PDB can't be used to check hydride/active-state.
    """
    tag = os.path.basename(os.path.normpath(d))
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(d, "motif.pdb"),
              os.path.join(os.environ.get("REPO_DIR", "/resnick/scratch/atiwari2/rfdiffusion"),
                           "pipeline/compiled", tag, "motif.pdb"),
              os.path.join(here, "compiled", tag, "motif.pdb"),
              os.path.join(here, "..", "pipeline/compiled", tag, "motif.pdb")):
        if os.path.isfile(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design_dir", nargs="?")
    ap.add_argument("--rules", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "v_chem_rules.yaml"))
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs="+", help="multiple design or compiled-motif dirs — tabulate gate-pass per target")
    args = ap.parse_args()

    rules = yaml.safe_load(open(args.rules))
    templates = rules["templates"]
    mapping = rules["target_to_template"]

    def score_dir(d):
        tag = os.path.basename(os.path.normpath(d))
        tname = mapping.get(tag)
        if not tname:
            raise SystemExit(f"no template mapping for target tag '{tag}' in {args.rules}")
        motif = resolve_motif_pdb(d)
        if not motif:
            raise SystemExit(f"motif.pdb not found for '{tag}' — looked in {d}, "
                             f"$REPO_DIR/pipeline/compiled/{tag}/, and pipeline/compiled/{tag}/")
        scores = [score_design(motif, templates[tname])]
        return tag, tname, scores

    if args.compare:
        rows = []
        for d in args.compare:
            tag, tname, scores = score_dir(d)
            rows.append(summarize(tag, scores, tname))
        cols = ("target", "n", "all_pass", "G_metal", "G_coord", "G_hapticity", "G_active_state")
        print(" | ".join(f"{c:>26}" if c == "target" else f"{c:>12}" for c in cols))
        for r in rows:
            g = r["frac_per_gate"]
            vals = (r["target"], r["n_designs"], r["frac_all_pass"],
                    g["G_metal"], g["G_coord"], g["G_hapticity"], g["G_active_state"])
            print(" | ".join(f"{str(v):>26}" if i == 0 else f"{str(v):>12}" for i, v in enumerate(vals)))
        # print one example failure reason per row for the gates that failed
        for r in rows:
            for gname in ("G_metal", "G_coord", "G_hapticity", "G_active_state"):
                if r["frac_per_gate"][gname] < 1.0 and r["example_failure_reason"][gname]:
                    print(f"   ! {r['target']} {gname}: {r['example_failure_reason'][gname]}")
        if args.out:
            json.dump(rows, open(args.out, "w"), indent=2)
        return

    if not args.design_dir:
        raise SystemExit("need <design_dir> or --compare")
    tag, tname, scores = score_dir(args.design_dir)
    summary = summarize(tag, scores, tname)
    out = args.out or os.path.join(args.design_dir, "scores_vchem.json")
    json.dump({"summary": summary, "designs": scores}, open(out, "w"), indent=2)
    print(f"=== V_chem v0 (template={tname}): {tag} ({len(scores)} designs) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
