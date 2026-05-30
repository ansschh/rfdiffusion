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


def derive_a_stereo_chem(cofactor_local):
    """A_stereo (DRAFT chemistry rule): directional bias for substrate approach face. For
    chiral metal-N,N complexes, donor asymmetry defines a preferred face. v_stereo = (N1->N2) x z
    points across the cone perpendicular to the chiral N,N axis. Soft preference for residues
    on the +v_stereo side. Disabled if <2 N donors or near-degenerate geometry.
    """
    n_donors = [c for c in cofactor_local if c["element"] == "N"]
    if len(n_donors) < 2:
        return None
    n1 = n_donors[0]["pos_local"]; n2 = n_donors[1]["pos_local"]
    v_n12 = (n2[0]-n1[0], n2[1]-n1[1], n2[2]-n1[2])
    # cross with +z axis (substrate-approach direction in local frame)
    cross = (v_n12[1]*1.0 - v_n12[2]*0.0,
             v_n12[2]*0.0 - v_n12[0]*1.0,
             v_n12[0]*0.0 - v_n12[1]*0.0)
    cn = math.sqrt(sum(c*c for c in cross))
    if cn < 0.01:
        return None
    v_stereo = [round(c/cn, 4) for c in cross]
    return {
        "v_stereo_local": v_stereo,
        "bias_strength": 0.3,
        "source_rule": "(N1->N2) x z (chiral N,N donor face asymmetry); applies inside substrate cone only",
    }


def derive_a_elec_chem(cofactor_local, manifest):
    """A_elec (DRAFT chemistry rule): for cationic-TS reactions (e.g., asymmetric transfer
    hydrogenation of imines), expect anionic-residue stabilization near the substrate-cone exit.
    Only emits when the target's reaction class is known to have a cationic TS.
    """
    pdb_id = (manifest.get("pdb_id") or "").upper()
    # cationic-TS classes; DRAFT lookup, expand as more targets are added.
    cationic_TS_targets = {"3ZP9", "5OD5"}    # ATH of cyclic imines (cationic iminium TS)
    if pdb_id not in cationic_TS_targets:
        return []
    h_local = next((c["pos_local"] for c in cofactor_local if c["element"] == "H"), None)
    if not h_local:
        return []
    return [{
        "type": "charged_acid",
        "mu_local": [round(h_local[0], 3), round(h_local[1], 3), round(h_local[2] + 4.0, 3)],
        "Sigma_diag": [2.0, 2.0, 2.0],
        "w": 0.5,
        "source_rule": "ATH cationic-TS stabilization: anionic residue (Asp/Glu) expected ~4 A "
                       "beyond hydride along substrate approach axis",
    }]


def derive_a_contact_chem(cofactor_local):
    """Chemistry-rule A_contact: derive Gaussians from COFACTOR GEOMETRY ALONE — no PDB residue coords.

    Rules (textbook-grounded, DRAFT — chemist calibration recommended):
      1. Cp/Cp* ring (5 C atoms at tight similar distance from metal): one hydrophobic
         Gaussian 3.5 A beyond the ring centroid, outward from metal.
      2. Each retained N donor: one polar Gaussian 3.0 A beyond, outward from metal.
      3. Each retained O donor (excluding cofactor-internal): one polar Gaussian, same.

    Sigma 1.5 A, w 1.0 (hydrophobic ring) / 0.7 (per-donor). No information from observed
    pocket residue coordinates is used — this is the non-oracle counterpart of the
    guidepost-derived A_contact.
    """
    a_contact = []

    # Rule 1: Cp/Cp* ring detection
    carbons = [c for c in cofactor_local if c["element"] == "C"]
    if len(carbons) >= 5:
        sorted_by_dist = sorted(carbons, key=lambda c: math.sqrt(sum(x*x for x in c["pos_local"])))
        ring = sorted_by_dist[:5]
        dists = [math.sqrt(sum(x*x for x in c["pos_local"])) for c in ring]
        spread = max(dists) - min(dists)
        if spread < 0.5:    # tight ring
            cx = sum(c["pos_local"][0] for c in ring) / 5
            cy = sum(c["pos_local"][1] for c in ring) / 5
            cz = sum(c["pos_local"][2] for c in ring) / 5
            cnorm = math.sqrt(cx*cx + cy*cy + cz*cz)
            if cnorm > 0.01:
                pos = [round(cx + 3.5 * cx/cnorm, 3),
                       round(cy + 3.5 * cy/cnorm, 3),
                       round(cz + 3.5 * cz/cnorm, 3)]
                a_contact.append({
                    "type": "hydrophobic", "mu_local": pos, "Sigma_diag": [1.5, 1.5, 1.5], "w": 1.0,
                    "source_rule": f"Cp-ring face: 5 C at {round(sum(dists)/5, 2)} A (spread {round(spread, 2)})",
                })

    # Rule 2: N donors -> polar Gaussian outward
    for c in cofactor_local:
        if c["element"] == "N":
            x, y, z = c["pos_local"]
            n = math.sqrt(x*x + y*y + z*z)
            if n > 0.5:
                pos = [round(x + 3.0 * x/n, 3), round(y + 3.0 * y/n, 3), round(z + 3.0 * z/n, 3)]
                a_contact.append({
                    "type": "polar", "mu_local": pos, "Sigma_diag": [1.5, 1.5, 1.5], "w": 0.7,
                    "source_rule": f"polar partner for N donor {c['name']}",
                })

    # Rule 3: O donors -> polar Gaussian outward
    for c in cofactor_local:
        if c["element"] == "O":
            x, y, z = c["pos_local"]
            n = math.sqrt(x*x + y*y + z*z)
            if n > 0.5:
                pos = [round(x + 3.0 * x/n, 3), round(y + 3.0 * y/n, 3), round(z + 3.0 * z/n, 3)]
                a_contact.append({
                    "type": "polar", "mu_local": pos, "Sigma_diag": [1.5, 1.5, 1.5], "w": 0.7,
                    "source_rule": f"polar partner for O donor {c['name']}",
                })

    return a_contact


def build_a_cat(target_dir, target_id, mode="oracle"):
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

    # A_contact: two modes.
    #   ORACLE: one typed Gaussian per curated guidepost residue, centered at the residue's
    #           sidechain centroid in the local frame (target-specific; uses observed pocket
    #           coords — privileged info; appropriate as an upper-bound diagnostic).
    #   CHEM:   Gaussians derived ONLY from cofactor geometry + reaction-class rules
    #           (no PDB residue coords; chemistry-only; the non-oracle baseline).
    if mode == "chem":
        a_contact = derive_a_contact_chem(cofactor_local)
    else:
        RESIDUE_TYPE = {
            "ALA":"hydrophobic","VAL":"hydrophobic","LEU":"hydrophobic","ILE":"hydrophobic",
            "MET":"hydrophobic","PRO":"hydrophobic","GLY":"small",
            "PHE":"aromatic","TYR":"aromatic","TRP":"aromatic",
            "SER":"polar","THR":"polar","ASN":"polar","GLN":"polar","CYS":"polar",
            "HIS":"anchor","LYS":"charged_base","ARG":"charged_base",
            "ASP":"charged_acid","GLU":"charged_acid",
        }
        BACKBONE = {"N","CA","C","O","OXT","H"}
        a_contact = []
        for gp in manifest.get("guideposts", []):
            chain = gp.get("chain"); resseq = gp.get("resseq"); resname = gp.get("resname", "")
            if not chain or resseq is None:
                continue
            sc_atoms = [a for a in atoms if a["chain"] == chain and a["resseq"] == resseq
                        and a["element"] != "H" and a["name"] not in BACKBONE]
            if not sc_atoms:
                continue
            n = len(sc_atoms)
            cx = sum(a["x"] for a in sc_atoms) / n
            cy = sum(a["y"] for a in sc_atoms) / n
            cz = sum(a["z"] for a in sc_atoms) / n
            mu = world_to_local((cx, cy, cz), origin, R)
            a_contact.append({
                "type": RESIDUE_TYPE.get(resname, "other"),
                "mu_local": mu,
                "Sigma_diag": [1.2, 1.2, 1.2],
                "w": 1.0,
                "source_residue": f"{resname}{resseq}{chain}",
                "min_dist_to_core": gp.get("min_dist_to_core"),
            })

    a_anchor = []
    if manifest.get("anchor", {}).get("treat_as") == "diagnostic":
        a_anchor.append({"type": manifest["anchor"].get("type"), "carried": False,
                         "status": "diagnostic (Rev1 — de novo need not reuse)"})

    a_ts = []
    if manifest.get("substrate", {}).get("name"):
        a_ts.append({"component": "substrate_reactive_atom", "pos_local": [0.0, 0.0, 3.0],
                     "sigma_A": 0.7, "confidence": "low",
                     "note": manifest["substrate"].get("pose_source", "transferred g_dd")})

    # New chem-mode-only channels (DRAFT rules — calibrate with PI):
    a_stereo = derive_a_stereo_chem(cofactor_local) if mode == "chem" else None
    a_elec_list = derive_a_elec_chem(cofactor_local, manifest) if mode == "chem" else []
    if not a_elec_list and mode != "chem":
        a_elec_payload = {"status": "missing", "would_carry": ["cationic_TS", "metal_charge", "dipoles"]}
    else:
        a_elec_payload = a_elec_list

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
            "A_stereo": a_stereo,
            "A_elec": a_elec_payload,
        },
        "uncertainty": {
            "A_TS": "low (transferred g_dd analog)",
            "A_anchor": "diagnostic (not load-bearing)" if mode == "oracle" else "draft chem rule",
            "A_stereo": "draft chem rule" if a_stereo else "missing",
            "A_elec": "draft chem rule" if a_elec_list else "missing",
            "A_dynamics": "missing", "A_solvent": "missing",
        },
        "cofactor_atoms_local": cofactor_local,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target_dir")
    ap.add_argument("--mode", choices=["oracle", "chem"], default="oracle",
                    help="oracle = A_contact from curated guidepost residues (uses observed pocket coords); "
                         "chem = A_contact from cofactor geometry + reaction-class rules only (no observed pocket coords)")
    ap.add_argument("--out")
    args = ap.parse_args()
    target_id = os.path.basename(os.path.normpath(args.target_dir))
    a_cat = build_a_cat(args.target_dir, target_id, mode=args.mode)
    a_cat["mode"] = args.mode
    default_out = f"A_cat_{args.mode}.json" if args.mode != "oracle" else "A_cat.json"
    out = args.out or os.path.join(args.target_dir, default_out)
    json.dump(a_cat, open(out, "w"), indent=2)
    print(f"=== A_cat instantiated for {target_id} (mode={args.mode}) ===")
    print(f"  cofactor atoms (local):  {len(a_cat['cofactor_atoms_local'])}")
    print(f"  A_steric components:     {len(a_cat['channels']['A_steric'])}")
    print(f"  A_contact components:    {len(a_cat['channels']['A_contact'])}")
    for c in a_cat['channels']['A_contact']:
        src = c.get('source_residue') or c.get('source_rule', '')
        print(f"      [{c['type']}]  mu={c['mu_local']}  Sigma={c['Sigma_diag']}  w={c['w']}  ({src})")
    print(f"  A_path:                  {'yes' if a_cat['channels']['A_path'] else 'no'}")
    print(f"  A_TS components:         {len(a_cat['channels']['A_TS'])}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
