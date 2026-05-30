#!/usr/bin/env python3
"""retrieve.py — rank candidate pockets against an A_cat query (gate-then-soft).

Score (per candidate, best across sampled orientations):
  gates (hard)     : G_clash (max steric overlap allowed),
                     G_path  (path-cone occupancy <= max_path_residues)
  soft (additive)  : Σ_τ w_τ Σ_r in pocket exp(-½(local_r - μ)^T Σ^-1 (local_r - μ))   (typed contact)
  soft (penalty)   : λ_clash * Σ residues inside any A_steric sphere   (overlap measure)
                     λ_path  * Σ residues inside A_path cone

Rotational search: spherical-Fibonacci sampling of N orientations (default 72).

Usage:
  python retrieve.py --acat <A_cat.json> --pockets <p1.json> [<p2.json> ...] [--n-rot 72]
"""
from __future__ import annotations
import argparse, json, math, os


def vsub(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def vdot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def vcross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def vlen(v): return math.sqrt(vdot(v,v))
def vnorm(v):
    n = vlen(v); return (v[0]/n, v[1]/n, v[2]/n) if n>0 else v
def vscale(v,s): return (v[0]*s, v[1]*s, v[2]*s)


def world_to_local_via_R(world_xyz, metal_xyz, R):
    """v_local = R @ (world - metal). R rows are local axes in world frame."""
    rel = vsub(world_xyz, metal_xyz)
    return (R[0][0]*rel[0]+R[0][1]*rel[1]+R[0][2]*rel[2],
            R[1][0]*rel[0]+R[1][1]*rel[1]+R[1][2]*rel[2],
            R[2][0]*rel[0]+R[2][1]*rel[1]+R[2][2]*rel[2])


def sample_rotations(n_target):
    """Spherical-Fibonacci z-axis sampling × 8 azimuthal rotations each."""
    n_z = max(1, n_target // 8)
    rotations = []
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    for i in range(n_z):
        z = 1.0 - 2.0 * (i + 0.5) / n_z
        r = math.sqrt(max(0.0, 1.0 - z*z))
        theta = 2.0 * math.pi * i / phi
        z_axis = (r*math.cos(theta), r*math.sin(theta), z)
        tmp = (0.0, 0.0, 1.0) if abs(z_axis[2]) < 0.99 else (1.0, 0.0, 0.0)
        x_raw = vsub(tmp, vscale(z_axis, vdot(tmp, z_axis)))
        x_axis = vnorm(x_raw)
        y_axis = vcross(z_axis, x_axis)
        for j in range(8):
            angle = 2.0 * math.pi * j / 8.0
            ca, sa = math.cos(angle), math.sin(angle)
            x_rot = (x_axis[0]*ca + y_axis[0]*sa, x_axis[1]*ca + y_axis[1]*sa, x_axis[2]*ca + y_axis[2]*sa)
            y_rot = (-x_axis[0]*sa + y_axis[0]*ca, -x_axis[1]*sa + y_axis[1]*ca, -x_axis[2]*sa + y_axis[2]*ca)
            rotations.append([[x_rot[0], x_rot[1], x_rot[2]],
                              [y_rot[0], y_rot[1], y_rot[2]],
                              [z_axis[0], z_axis[1], z_axis[2]]])
    return rotations


def score_orientation(a_cat, pocket, R, params):
    metal_xyz = pocket["metal"]["world"]
    channels = a_cat["channels"]
    include = params["include"]
    contacts = channels.get("A_contact", []) if "A_contact" in include else []
    sterics  = channels.get("A_steric", [])  if "A_steric"  in include else []
    path     = channels.get("A_path")        if "A_path"    in include else None
    stereo   = channels.get("A_stereo")      if "A_stereo"  in include else None
    elec     = channels.get("A_elec", [])    if "A_elec"    in include else []
    if isinstance(elec, dict):
        elec = []   # legacy "missing" stub — empty list
    anchor   = channels.get("A_anchor", [])  if "A_anchor"  in include else []

    # Pre-transform pocket residues to local frame (once per orientation).
    locals_by_type = {}
    for res in pocket["pocket_residues"]:
        local = world_to_local_via_R(res["sidechain_centroid_world"], metal_xyz, R)
        locals_by_type.setdefault(res["type"], []).append((local, res))

    # Winner-take-all per Gaussian: each A_contact picks its BEST matching residue (by type +
    # position). Bounds score by sum_of_weights, removes pocket-size bias.
    score = 0.0
    matches = []
    for c in contacts:
        best_o = 0.0; best_res = None
        for (local, res) in locals_by_type.get(c["type"], []):
            mu = c["mu_local"]; sig = c["Sigma_diag"]
            d2 = ((local[0]-mu[0])/sig[0])**2 + ((local[1]-mu[1])/sig[1])**2 + ((local[2]-mu[2])/sig[2])**2
            o = math.exp(-0.5 * d2)
            if o > best_o:
                best_o = o; best_res = f"{res['resname']}{res['resseq']}{res['chain']}"
        score += c["w"] * best_o
        matches.append({"gaussian_type": c["type"], "best_overlap": round(best_o, 3),
                        "best_residue": best_res, "source": c.get("source_residue")})

    # Clash with A_steric exclusion spheres + path-cone occupancy (iterate all residues once).
    clash = 0.0
    path_res = 0
    for type_list in locals_by_type.values():
        for (local, _res) in type_list:
            for s in sterics:
                p = s["pos_local"]; r = s["r"]
                d = math.sqrt((local[0]-p[0])**2 + (local[1]-p[1])**2 + (local[2]-p[2])**2)
                if d < r:
                    clash += (r - d) ** 2
            if path:
                apex = path["apex_local"]; axis = tuple(path["axis_local"])
                rel = vsub(local, apex)
                proj = vdot(rel, axis)
                if 0.3 < proj < path["extent_A"]:
                    perp2 = max(0.0, vdot(rel, rel) - proj*proj)
                    if math.sqrt(perp2) < proj * math.tan(math.radians(path["half_angle_deg"])):
                        path_res += 1

    # A_anchor scoring: winner-take-all per anchor Gaussian (same shape as A_contact).
    anchor_score = 0.0
    for a in anchor:
        if not a.get("carried", True):
            continue   # skip diagnostic-only anchor entries
        atype = a.get("type", "anchor"); mu = a.get("mu_local"); sig = a.get("Sigma_diag", [1.5,1.5,1.5])
        if not mu:
            continue
        best_o = 0.0
        for (local, _res) in locals_by_type.get(atype, []):
            d2 = ((local[0]-mu[0])/sig[0])**2 + ((local[1]-mu[1])/sig[1])**2 + ((local[2]-mu[2])/sig[2])**2
            o = math.exp(-0.5 * d2)
            best_o = max(best_o, o)
        anchor_score += a.get("w", 0.5) * best_o

    # A_stereo scoring: residues on the +v_stereo side of substrate cone get a small bonus.
    stereo_score = 0.0
    if stereo and path:
        v = stereo.get("v_stereo_local", [0,0,0]); bias = stereo.get("bias_strength", 0.0)
        apex = path["apex_local"]; axis = tuple(path["axis_local"])
        for type_list in locals_by_type.values():
            for (local, _res) in type_list:
                rel = vsub(local, apex)
                proj = vdot(rel, axis)
                if 0.3 < proj < path["extent_A"]:
                    # residue is in the substrate cone region
                    side = rel[0]*v[0] + rel[1]*v[1] + rel[2]*v[2]
                    if side > 0:
                        stereo_score += bias

    # A_elec scoring: typed Gaussian against charged residues (same shape as A_contact).
    elec_score = 0.0
    for e in elec:
        etype = e.get("type", "charged_acid"); mu = e.get("mu_local"); sig = e.get("Sigma_diag", [2.0,2.0,2.0])
        if not mu:
            continue
        best_o = 0.0
        for (local, _res) in locals_by_type.get(etype, []):
            d2 = ((local[0]-mu[0])/sig[0])**2 + ((local[1]-mu[1])/sig[1])**2 + ((local[2]-mu[2])/sig[2])**2
            o = math.exp(-0.5 * d2)
            best_o = max(best_o, o)
        elec_score += e.get("w", 0.5) * best_o

    soft = (score + anchor_score + stereo_score + elec_score
            - params["lambda_clash"] * clash - params["lambda_path"] * path_res)
    gate_clash_ok = clash <= params["max_clash_overlap"]
    gate_path_ok = path_res <= params["max_path_residues"]
    return {
        "soft_score": soft, "contact_score": round(score, 3),
        "anchor_score": round(anchor_score, 3), "stereo_score": round(stereo_score, 3),
        "elec_score": round(elec_score, 3),
        "clash_overlap": round(clash, 3), "n_path_residues": path_res,
        "gates_pass": gate_clash_ok and gate_path_ok,
    }


def score_pocket(a_cat, pocket, params, rotations):
    best = None
    for R in rotations:
        r = score_orientation(a_cat, pocket, R, params)
        if best is None or r["soft_score"] > best["soft_score"]:
            best = r
            best["R"] = R
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acat", required=True, help="A_cat JSON (from instantiate_acat.py)")
    ap.add_argument("--pockets", nargs="+", required=True, help="Pocket JSONs (from extract_pocket.py)")
    ap.add_argument("--n-rot", type=int, default=144)
    ap.add_argument("--lambda-clash", type=float, default=1.0)
    ap.add_argument("--lambda-path", type=float, default=2.0)
    ap.add_argument("--max-clash-overlap", type=float, default=3.0)
    ap.add_argument("--max-path-residues", type=int, default=1)
    ap.add_argument("--include", nargs="+",
                    default=["A_contact", "A_path", "A_steric"],
                    help="Channels to use for scoring (ablation knob). Subset of "
                         "A_contact / A_path / A_steric / A_anchor / A_stereo / A_elec.")
    ap.add_argument("--save-best-R", help="If set, save per-pocket best_R rotation matrices to this JSON "
                                           "(needed by transplant_pocket.py for the heme-as-scaffold experiment)")
    ap.add_argument("--out")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    a_cat = json.load(open(args.acat))
    params = {"lambda_clash": args.lambda_clash, "lambda_path": args.lambda_path,
              "max_clash_overlap": args.max_clash_overlap, "max_path_residues": args.max_path_residues,
              "include": set(args.include)}
    rotations = sample_rotations(args.n_rot)

    results = []
    best_R_by_pocket = {}
    for ppath in args.pockets:
        pocket = json.load(open(ppath))
        best = score_pocket(a_cat, pocket, params, rotations)
        results.append({
            "pdb_id": pocket.get("pdb_id"),
            "metal": pocket.get("metal", {}).get("element"),
            "n_residues": pocket.get("n_residues"),
            "soft_score": round(best["soft_score"], 3),
            "contact_score": best["contact_score"],
            "anchor_score": best.get("anchor_score", 0.0),
            "stereo_score": best.get("stereo_score", 0.0),
            "elec_score": best.get("elec_score", 0.0),
            "clash_overlap": best["clash_overlap"],
            "n_path_residues": best["n_path_residues"],
            "gates_pass": best["gates_pass"],
            "pocket_file": ppath,
        })
        best_R_by_pocket[pocket.get("pdb_id")] = {
            "best_R": best["R"],
            "pocket_file": ppath,
            "metal_world": pocket["metal"]["world"],
        }
    results.sort(key=lambda r: r["soft_score"], reverse=True)

    if args.save_best_R:
        json.dump(best_R_by_pocket, open(args.save_best_R, "w"), indent=2)
        print(f"  saved best_R per pocket -> {args.save_best_R}")

    print(f"=== retrieval query: {a_cat['target']}  ({len(args.pockets)} candidates, {args.n_rot} orientations) ===")
    print(f"    channels included: {sorted(args.include)}")
    cols = ("rank", "pdb_id", "metal", "n_res", "soft", "contact", "anchor", "stereo", "elec", "path_res", "gates")
    print(" | ".join(f"{c:>9}" for c in cols))
    for i, r in enumerate(results[:args.top]):
        vals = (i+1, r["pdb_id"], r["metal"], r["n_residues"], r["soft_score"],
                r["contact_score"], r["anchor_score"], r["stereo_score"], r["elec_score"],
                r["n_path_residues"], "ok" if r["gates_pass"] else "FAIL")
        print(" | ".join(f"{str(v):>9}" for v in vals))
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
