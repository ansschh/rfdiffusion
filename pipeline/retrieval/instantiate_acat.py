#!/usr/bin/env python3
"""instantiate_acat.py — emit a concrete A_cat JSON from a compiled motif.

Reads <target_dir>/motif.pdb + manifest.json, computes the reactive-core local frame
(origin=metal; +z=metal->open-site atom; +x=projected metal->N-donor-centroid), and
emits A_cat channels (A_steric, A_contact, A_path, A_anchor, A_TS, A_elec=missing).

Usage:
  python instantiate_acat.py pipeline/compiled/3ZP9
"""
from __future__ import annotations
import argparse, json, math, os

METALS = {"IR","ZN","RH","RU","FE","MN","CU","CO","NI","PD","PT","MO","W","OS","V","CR","MG","CA","NA","K","AL"}
VDW = {"H":1.20,"C":1.70,"N":1.55,"O":1.52,"S":1.80,"F":1.47,"P":1.80,"CL":1.75,"BR":1.85,
       "IR":2.00,"RH":2.00,"RU":2.00,"ZN":1.95,"FE":1.95,"MG":1.73,"MN":1.95,"CU":1.95,"CO":1.95,"NI":1.95}


def vsub(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def vdot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def vcross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def vlen(v): return math.sqrt(vdot(v,v))
def vnorm(v):
    n = vlen(v); return (v[0]/n, v[1]/n, v[2]/n) if n>0 else v
def vscale(v,s): return (v[0]*s, v[1]*s, v[2]*s)


def parse_pdb(path):
    out = []
    for line in open(path):
        if line[:6].strip() not in ("ATOM","HETATM"): continue
        try:
            name = line[12:16].strip()
            el = line[76:78].strip().upper() or "".join(c for c in name if c.isalpha())[:2].upper()
            out.append({
                "record": line[:6].strip(), "name": name, "element": el,
                "resname": line[17:20].strip(), "chain": line[21].strip(),
                "resseq": int(line[22:26]),
                "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
            })
        except ValueError:
            continue
    return out


def find_metal(atoms):
    for a in atoms:
        if a["record"]=="HETATM" and a["resname"]!="ORI" and a["element"] in METALS:
            return a
    return None


def compute_frame(atoms):
    """origin = metal; +z = metal -> open-site (H/O1) or fallback opposite donor centroid;
    +x  = (projected) metal -> N-donor centroid."""
    metal = find_metal(atoms)
    if not metal: raise SystemExit("no metal in motif")
    origin = (metal["x"], metal["y"], metal["z"])
    h = next((a for a in atoms if a["resname"]=="LIG" and a["element"]=="H"), None)
    o1 = next((a for a in atoms if a["resname"]=="LIG" and a["name"]=="O1"), None)
    open_atom = h or o1
    if open_atom:
        v_open = vsub((open_atom["x"], open_atom["y"], open_atom["z"]), origin)
    else:
        donors = [a for a in atoms if a["resname"]=="LIG" and a["element"] in ("N","O","C") and a is not metal]
        cx = sum(a["x"] for a in donors)/len(donors)
        cy = sum(a["y"] for a in donors)/len(donors)
        cz = sum(a["z"] for a in donors)/len(donors)
        v_open = vsub(origin, (cx, cy, cz))
    z_axis = vnorm(v_open)
    ns = [a for a in atoms if a["resname"]=="LIG" and a["element"]=="N"]
    if ns:
        cx = sum(a["x"] for a in ns)/len(ns); cy = sum(a["y"] for a in ns)/len(ns); cz = sum(a["z"] for a in ns)/len(ns)
        v_x_raw = vsub((cx,cy,cz), origin)
    else:
        v_x_raw = (1.0, 0.0, 0.0) if abs(z_axis[0])<0.9 else (0.0, 1.0, 0.0)
    v_x_proj = vsub(v_x_raw, vscale(z_axis, vdot(v_x_raw, z_axis)))
    if vlen(v_x_proj) < 0.01:
        v_x_proj = vcross(z_axis, (0.0,1.0,0.0))
        if vlen(v_x_proj) < 0.01:
            v_x_proj = vcross(z_axis, (1.0,0.0,0.0))
    x_axis = vnorm(v_x_proj)
    y_axis = vcross(z_axis, x_axis)
    R = [list(x_axis), list(y_axis), list(z_axis)]    # rows = local axes in world frame
    return origin, R, metal, open_atom


def world_to_local(p, origin, R):
    rel = vsub(p, origin)
    return [round(R[i][0]*rel[0] + R[i][1]*rel[1] + R[i][2]*rel[2], 3) for i in range(3)]


def build_a_cat(target_dir, target_id):
    atoms = parse_pdb(os.path.join(target_dir, "motif.pdb"))
    manifest = json.load(open(os.path.join(target_dir, "manifest.json")))
    origin, R, metal, open_atom = compute_frame(atoms)

    cofactor_local = []
    for a in atoms:
        if a["resname"] == "LIG":
            lp = world_to_local((a["x"], a["y"], a["z"]), origin, R)
            cofactor_local.append({"name": a["name"], "element": a["element"], "pos_local": lp})

    a_steric = []
    for c in cofactor_local:
        r = 1.4 if c["element"]=="H" else VDW.get(c["element"], 1.7) + 0.4
        a_steric.append({"name": c["name"], "pos_local": c["pos_local"], "r": round(r, 2)})

    h_local = next((c["pos_local"] for c in cofactor_local if c["element"]=="H"), None)
    a_path = None
    if h_local:
        a_path = {"apex_local": h_local, "axis_local": [0.0, 0.0, 1.0],
                  "half_angle_deg": 30, "extent_A": 5.5,
                  "content_type": "substrate_reactive_atom",
                  "expected_M_substrate_A": [2.5, 3.5]}

    # ATH-style hydrophobic groove on the Cp* face (z > 0 side)
    a_contact = [
        {"type": "hydrophobic", "mu_local": [3.5, 0.0, 2.5], "Sigma_diag": [1.8, 1.8, 1.8], "w": 1.0,
         "note": "Cp*-face hydrophobic groove"},
        {"type": "hydrophobic", "mu_local": [-3.0, 2.0, 1.5], "Sigma_diag": [1.8, 1.8, 1.8], "w": 1.0,
         "note": "lateral hydrophobic"},
        {"type": "polar", "mu_local": [2.5, -3.0, -0.5], "Sigma_diag": [1.5, 1.5, 1.5], "w": 0.5,
         "note": "sulfonamide side / anchor-adjacent"},
    ]

    a_anchor = []
    if manifest.get("anchor", {}).get("treat_as") == "diagnostic":
        a_anchor.append({"type": manifest["anchor"].get("type"), "carried": False,
                         "status": "diagnostic (Rev1 — de novo need not reuse)"})

    a_ts = []
    if manifest.get("substrate", {}).get("name"):
        a_ts.append({"component": "substrate_reactive_atom", "pos_local": [0.0, 0.0, 3.0],
                     "sigma_A": 0.7, "confidence": "low",
                     "note": manifest["substrate"].get("pose_source", "transferred g_dd")})

    return {
        "target": target_id,
        "frame": {
            "origin_world": [round(c,3) for c in origin],
            "R_world_to_local": [[round(v,6) for v in row] for row in R],
            "axes_note": "rows of R = local axes expressed in world frame; v_local = R @ (v_world - origin)",
        },
        "channels": {
            "A_steric": a_steric,
            "A_contact": a_contact,
            "A_path": a_path,
            "A_anchor": a_anchor,
            "A_TS": a_ts,
            "A_elec": {"status": "missing", "would_carry": ["cationic_TS", "metal_charge", "dipoles"]},
        },
        "uncertainty": {
            "A_TS": "low (transferred g_dd analog)",
            "A_anchor": "diagnostic (not load-bearing)",
            "A_elec": "missing", "A_dynamics": "missing", "A_solvent": "missing",
        },
        "cofactor_atoms_local": cofactor_local,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target_dir")
    ap.add_argument("--out")
    args = ap.parse_args()
    target_id = os.path.basename(os.path.normpath(args.target_dir))
    a_cat = build_a_cat(args.target_dir, target_id)
    out = args.out or os.path.join(args.target_dir, "A_cat.json")
    json.dump(a_cat, open(out, "w"), indent=2)
    print(f"=== A_cat instantiated for {target_id} ===")
    print(f"  cofactor atoms (local):  {len(a_cat['cofactor_atoms_local'])}")
    print(f"  A_steric components:     {len(a_cat['channels']['A_steric'])}")
    print(f"  A_contact components:    {len(a_cat['channels']['A_contact'])}")
    print(f"  A_path:                  {'yes' if a_cat['channels']['A_path'] else 'no'}")
    print(f"  A_TS components:         {len(a_cat['channels']['A_TS'])}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
