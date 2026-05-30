#!/usr/bin/env python3
"""V_rxn v0 — reaction-geometry validation (Rev2 crit-3 / "open-cone accessibility").

For ATH-style cofactors (metal + hydride at the open coordination), checks whether the
substrate approach path is clear: a cone with apex at the metal, axis along metal->hydride,
half-angle ~30 deg, extending ~5.5 A BEYOND the hydride. The substrate (imine C) would
normally occupy this region; any protein/ligand heavy atom inside it blocks the chemistry.

Gates:
  G_access: hydride present at expected metal-H distance AND no heavy atom in the approach cone.

v0 scope: ATH cofactor (3ZP9-style — synthesized hydride). 5OD5 (His-coordinated) needs its
own reaction-class spec because His227 occupies the open leg in the natural enzyme.

Usage:
  python v_rxn.py <design_dir>
  python v_rxn.py --compare <dir1> <dir2> ...
"""
from __future__ import annotations
import argparse, glob, json, math, os

METALS = {"IR","ZN","RH","RU","FE","MN","CU","CO","NI","PD","PT","MO","W","OS","V","CR","MG","CA","NA","K","AL"}

# Defaults for ATH-style open-site (metal->hydride axis, substrate approach beyond).
ATH_HALF_ANGLE_DEG = 30.0
ATH_CONE_BEYOND = 5.5         # how far beyond the hydride to check
ATH_MH_DIST_BAND = (1.40, 1.85)
ATH_MAX_BLOCKERS_PASS = 0     # categorical: any blocker -> FAIL


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
    cands = [a for a in atoms if a[0] == "HETATM" and a[3] != "ORI" and a[2] in METALS]
    if not cands:
        cands = [a for a in atoms if a[2] in METALS]
    return cands[0] if cands else None


def find_open_atom(metal, atoms, element="H", max_d=2.5):
    """Closest atom of given element (e.g., synthesized hydride) within max_d of the metal."""
    m = (metal[4], metal[5], metal[6])
    best, best_d = None, max_d + 1
    for a in atoms:
        if a is metal or a[2] != element:
            continue
        d = dist(m, (a[4], a[5], a[6]))
        if d < best_d:
            best, best_d = a, d
    return best, best_d


def count_cone_blockers(metal, open_atom, atoms, half_angle_deg, cone_beyond):
    """Heavy atoms inside the substrate-approach cone (apex=metal, axis=metal->open_atom,
    region from just-beyond-open_atom out to open_atom + cone_beyond)."""
    m = (metal[4], metal[5], metal[6])
    o = (open_atom[4], open_atom[5], open_atom[6])
    ax = (o[0]-m[0], o[1]-m[1], o[2]-m[2])
    n = math.sqrt(ax[0]**2 + ax[1]**2 + ax[2]**2)
    if n == 0:
        return 0, []
    axd = (ax[0]/n, ax[1]/n, ax[2]/n)
    proj_open = n
    max_extent = proj_open + cone_beyond
    tan_h = math.tan(math.radians(half_angle_deg))
    blockers = []
    for a in atoms:
        if a is metal or a is open_atom or a[2] == "H" or a[3] == "ORI":
            continue
        rel = (a[4]-m[0], a[5]-m[1], a[6]-m[2])
        proj = rel[0]*axd[0] + rel[1]*axd[1] + rel[2]*axd[2]
        if proj <= proj_open + 0.3 or proj > max_extent:
            continue
        rel2 = rel[0]**2 + rel[1]**2 + rel[2]**2
        perp = math.sqrt(max(0.0, rel2 - proj**2))
        if perp <= proj * tan_h:
            blockers.append({"name": a[1], "el": a[2], "resname": a[3],
                             "proj_A": round(proj, 2), "perp_A": round(perp, 2)})
    return len(blockers), blockers


def score_design(path, params):
    atoms = parse_design(path)
    metal = find_metal(atoms)
    if not metal:
        return {"file": os.path.basename(path), "G_access": {"pass": False, "reason": "no metal found"}, "all_pass": False}
    open_atom, mh = find_open_atom(metal, atoms, element="H")
    if not open_atom:
        return {"file": os.path.basename(path), "metal": metal[2],
                "G_access": {"pass": False, "reason": "no hydride at open site (required for ATH V_rxn)"},
                "all_pass": False}
    lo, hi = params["mh_dist_band"]
    if not (lo <= mh <= hi):
        return {"file": os.path.basename(path), "metal": metal[2], "mh_dist": round(mh, 3),
                "G_access": {"pass": False, "reason": f"metal-H distance {round(mh,3)} A outside band [{lo},{hi}]"},
                "all_pass": False}
    n_block, blockers = count_cone_blockers(metal, open_atom, atoms,
                                            params["half_angle_deg"], params["cone_beyond"])
    if n_block <= params["max_blockers_pass"]:
        ga = {"pass": True, "n_cone_blockers": n_block, "blockers": blockers}
    else:
        ga = {"pass": False, "n_cone_blockers": n_block, "blockers": blockers,
              "reason": f"{n_block} heavy atom(s) inside substrate-approach cone"}
    return {"file": os.path.basename(path), "metal": metal[2],
            "mh_dist": round(mh, 3), "G_access": ga, "all_pass": ga["pass"]}


def list_pdbs(d):
    pdbs = sorted(p for p in glob.glob(os.path.join(d, "*-atomized-bb-False.pdb"))
                  if "/unidealized/" not in p.replace("\\", "/"))
    if not pdbs:
        m = os.path.join(d, "motif.pdb")
        if os.path.isfile(m):
            pdbs = [m]
    return pdbs


def summarize(target, scores):
    n = len(scores)
    if n == 0:
        return {"target": target, "n_designs": 0}
    pass_n = sum(1 for s in scores if s.get("all_pass"))
    cone_counts = [s["G_access"].get("n_cone_blockers", 0) for s in scores if "G_access" in s]
    return {
        "target": target, "n_designs": n,
        "frac_access_pass": round(pass_n / n, 3),
        "median_cone_blockers": sorted(cone_counts)[len(cone_counts)//2] if cone_counts else None,
        "first_fail_reason": next((s["G_access"].get("reason") for s in scores
                                    if not s.get("all_pass") and "G_access" in s), None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design_dir", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--half-angle", type=float, default=ATH_HALF_ANGLE_DEG)
    ap.add_argument("--cone-beyond", type=float, default=ATH_CONE_BEYOND)
    ap.add_argument("--max-blockers", type=int, default=ATH_MAX_BLOCKERS_PASS)
    ap.add_argument("--mh-band", nargs=2, type=float, default=list(ATH_MH_DIST_BAND))
    ap.add_argument("--compare", nargs="+")
    args = ap.parse_args()
    params = {"half_angle_deg": args.half_angle, "cone_beyond": args.cone_beyond,
              "max_blockers_pass": args.max_blockers, "mh_dist_band": tuple(args.mh_band)}

    def score_dir(d):
        tag = os.path.basename(os.path.normpath(d))
        scores = [score_design(p, params) for p in list_pdbs(d)]
        return tag, scores

    if args.compare:
        rows = [summarize(*score_dir(d)) for d in args.compare]
        cols = ("target", "n", "frac_access_pass", "median_cone_blockers")
        print(" | ".join(f"{c:>26}" if c == "target" else f"{c:>18}" for c in cols))
        for r in rows:
            vals = (r["target"], r["n_designs"], r.get("frac_access_pass"), r.get("median_cone_blockers"))
            print(" | ".join(f"{str(v):>26}" if i == 0 else f"{str(v):>18}" for i, v in enumerate(vals)))
        for r in rows:
            if r.get("first_fail_reason"):
                print(f"   ! {r['target']}: {r['first_fail_reason']}")
        if args.out:
            json.dump(rows, open(args.out, "w"), indent=2)
        return

    if not args.design_dir:
        raise SystemExit("need <design_dir> or --compare")
    tag, scores = score_dir(args.design_dir)
    summary = summarize(tag, scores)
    out = args.out or os.path.join(args.design_dir, "scores_vrxn.json")
    json.dump({"summary": summary, "designs": scores}, open(out, "w"), indent=2)
    print(f"=== V_rxn v0 (ATH open-cone): {tag} ({len(scores)} designs) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
