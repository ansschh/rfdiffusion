#!/usr/bin/env python3
"""damaged_controls.py - generate damaged A_cat variants for discriminativity
tests. Each variant is a deep copy of the input A_cat with ONE channel altered
in a chemically meaningful way.

Damage taxonomy (PI's list + extensions):

  rotated_path_90      - rotate A_path.axis_local by 90 deg about +y (lateral)
  rotated_path_180     - rotate axis by 180 (cone now points backward into M)
  inverted_face        - flip the entire local frame (z -> -z) i.e. inverted
                         proximal/distal. Realized by inverting axis_local and
                         negating mu_local z component of all contact Gaussians.
  wrong_hapticity      - inverts contact Gaussian types: aromatic <-> charged_acid,
                         hydrophobic <-> charged_base, mimicking "the cofactor
                         actually presents kappa-O,O not kappa-N,N type chemistry"
  blocked_open_site    - shift A_path apex 2 A in +z (deep into protein); cone
                         now points into a region the protein normally fills
  rotated_contact_90   - rotate all A_contact Gaussians 90 deg about z (axial)
  shuffled_contact     - randomize A_contact Gaussian positions within a 8 A
                         cube around metal (seeded)
  null_acat            - empty all channels (catalytic facts removed entirely)

Output: a dict { name -> damaged A_cat dict }. Each retains the original frame
and cofactor_atoms_local so the same protein can be re-scored under multiple
damages.
"""
from __future__ import annotations
import argparse, copy, json, math, os, random


def _rotate_3(v, axis, deg):
    """Rotate vector v about axis by deg degrees (right-hand). Axis must be unit."""
    th = math.radians(deg); c = math.cos(th); s = math.sin(th)
    ax, ay, az = axis
    dot = v[0]*ax + v[1]*ay + v[2]*az
    return (
        v[0]*c + (ay*v[2]-az*v[1])*s + ax*dot*(1-c),
        v[1]*c + (az*v[0]-ax*v[2])*s + ay*dot*(1-c),
        v[2]*c + (ax*v[1]-ay*v[0])*s + az*dot*(1-c),
    )


def damage_rotated_path(a_cat, deg=90, axis=(0.0, 1.0, 0.0)):
    new = copy.deepcopy(a_cat)
    if new["channels"].get("A_path"):
        ax = new["channels"]["A_path"].get("axis_local", [0.0, 0.0, 1.0])
        rotated = _rotate_3(tuple(ax), axis, deg)
        new["channels"]["A_path"]["axis_local"] = [round(c, 4) for c in rotated]
        # also rotate apex_local (the hydride position in local frame)
        ap = new["channels"]["A_path"].get("apex_local")
        if ap:
            rotated_ap = _rotate_3(tuple(ap), axis, deg)
            new["channels"]["A_path"]["apex_local"] = [round(c, 3) for c in rotated_ap]
    new["_damage"] = f"rotated_path_{deg}_about_{axis}"
    return new


def damage_inverted_face(a_cat):
    """z -> -z: invert proximal/distal across ALL channels. The 2026-05-31 fix
    extends to A_face (was missing in original implementation, which led to the
    inverted_face=0 artifact in oracle Stage 2c) and A_elec_field."""
    new = copy.deepcopy(a_cat)
    ch = new["channels"]
    if ch.get("A_path"):
        ap = ch["A_path"].get("apex_local")
        if ap: ap[2] = -ap[2]
        ax = ch["A_path"].get("axis_local")
        if ax: ax[2] = -ax[2]
    for g in (ch.get("A_contact") or []):
        g["mu_local"][2] = -g["mu_local"][2]
    for s in (ch.get("A_steric") or []):
        s["pos_local"][2] = -s["pos_local"][2]
    for t in (ch.get("A_TS") or []):
        if "pos_local" in t: t["pos_local"][2] = -t["pos_local"][2]
    for e in (ch.get("A_elec") or []) if isinstance(ch.get("A_elec"), list) else []:
        if "mu_local" in e: e["mu_local"][2] = -e["mu_local"][2]
    # A_face: flip the reactive axis z component
    if ch.get("A_face"):
        for k in ("reactive_axis_local", "packing_axis_local"):
            if k in ch["A_face"]:
                ch["A_face"][k][2] = -ch["A_face"][k][2]
    # A_elec_field: flip ts_center z + dipole direction z
    if ch.get("A_elec_field"):
        ef = ch["A_elec_field"]
        if "ts_center_local" in ef: ef["ts_center_local"][2] = -ef["ts_center_local"][2]
        if "ts_dipole_direction_local" in ef: ef["ts_dipole_direction_local"][2] = -ef["ts_dipole_direction_local"][2]
    # A_coord_zones: flip z of each zone
    for z in (ch.get("A_coord_zones") or []):
        if "center_local" in z: z["center_local"][2] = -z["center_local"][2]
    new["_damage"] = "inverted_face_z_flip_FULL (2026-05-31)"
    return new


def damage_wrong_hapticity(a_cat):
    """Swap contact Gaussian types so the chemistry asks for fundamentally
    different residue classes."""
    new = copy.deepcopy(a_cat)
    swap = {
        "hydrophobic": "charged_base",
        "aromatic":    "charged_acid",
        "polar":       "hydrophobic",
        "anchor":      "small",
        "charged_acid": "hydrophobic",
        "charged_base": "hydrophobic",
        "small":       "anchor",
    }
    for g in (new["channels"].get("A_contact") or []):
        g["type"] = swap.get(g.get("type"), "small")
    new["_damage"] = "wrong_hapticity_type_swap"
    return new


def damage_blocked_open_site(a_cat, tilt_deg=60, axis=(0.0, 1.0, 0.0)):
    """Simulates 'the open face is occluded': tilt cone axis by tilt_deg
    laterally (into the protein wall) AND inflate cone by 50% so it grabs more
    of the surrounding protein. This is the continuous analog of the
    motif_compiler block_open_site damage (planting a steric blocker)."""
    new = copy.deepcopy(a_cat)
    ap = new["channels"].get("A_path")
    if ap:
        ax = ap.get("axis_local", [0.0, 0.0, 1.0])
        rotated = _rotate_3(tuple(ax), axis, tilt_deg)
        ap["axis_local"] = [round(c, 4) for c in rotated]
        ap["half_angle_deg"] = ap.get("half_angle_deg", 30.0) * 1.3
    new["_damage"] = f"blocked_open_site_tilt_{tilt_deg}_widen_30pc"
    return new


def damage_rotated_contact(a_cat, deg=90, axis=(0.0, 0.0, 1.0)):
    new = copy.deepcopy(a_cat)
    for g in (new["channels"].get("A_contact") or []):
        rotated = _rotate_3(tuple(g["mu_local"]), axis, deg)
        g["mu_local"] = [round(c, 3) for c in rotated]
    new["_damage"] = f"rotated_contact_{deg}_about_{axis}"
    return new


def damage_shuffled_contact(a_cat, seed=42, span_A=8.0):
    new = copy.deepcopy(a_cat)
    rng = random.Random(seed)
    for g in (new["channels"].get("A_contact") or []):
        g["mu_local"] = [round(rng.uniform(-span_A, span_A), 3) for _ in range(3)]
    new["_damage"] = f"shuffled_contact_seed_{seed}"
    return new


def damage_flip_face(a_cat):
    """Flip the proximal/distal axis: reactive face becomes packing face and
    vice versa. Simulates the 2CCY heme-transplant failure mode: cofactor
    asymmetry was misidentified, residues placed on the WRONG side."""
    new = copy.deepcopy(a_cat)
    face = new["channels"].get("A_face")
    if face:
        ax = face.get("reactive_axis_local", [0.0, 0.0, 1.0])
        face["reactive_axis_local"] = [-ax[0], -ax[1], -ax[2]]
        pa = face.get("packing_axis_local", [0.0, 0.0, -1.0])
        face["packing_axis_local"] = [-pa[0], -pa[1], -pa[2]]
    new["_damage"] = "flipped_face_axis (proximal/distal swap)"
    return new


def damage_remove_coord_zones(a_cat):
    """Remove all allowed coord zones - now any donor near metal is poison
    (E_coord_zones harshly penalizes cofactor donor regions too). Bad damage
    on a real protein - the cofactor's own retained donors trip the term."""
    new = copy.deepcopy(a_cat)
    new["channels"]["A_coord_zones"] = []
    new["_damage"] = "removed_all_coord_zones"
    return new


def damage_shift_coord_zones(a_cat, shift_A=3.0):
    """Shift coord zones AWAY from cofactor donors. Now cofactor's own donors
    are outside any zone -> E_coord_zones won't help, AND any protein donor
    that happens to be near the shifted (wrong) zones is exempted incorrectly."""
    new = copy.deepcopy(a_cat)
    for z in (new["channels"].get("A_coord_zones") or []):
        c = z.get("center_local")
        if c is not None:
            c[0] += shift_A
    new["_damage"] = f"shifted_coord_zones_x_{shift_A}A"
    return new


def damage_flip_elec_dipole(a_cat):
    """Flip the TS dipole direction: reward becomes penalty.

    Effect on E_elec_field: any field that was stabilizing the TS dipole now
    points opposite to it -> destabilizing. Real charged residues will see
    their sign of contribution flipped.

    Tests whether E_elec_field is doing real chemistry-direction work."""
    new = copy.deepcopy(a_cat)
    ef = new["channels"].get("A_elec_field")
    if ef and "ts_dipole_direction_local" in ef:
        d = ef["ts_dipole_direction_local"]
        ef["ts_dipole_direction_local"] = [-d[0], -d[1], -d[2]]
    new["_damage"] = "flipped_elec_dipole_direction"
    return new


def damage_remove_elec_field(a_cat):
    """Strip A_elec_field entirely - E_elec_field returns 0. Tests whether
    presence of the channel itself produces signal (vs default zero)."""
    new = copy.deepcopy(a_cat)
    new["channels"]["A_elec_field"] = None
    new["_damage"] = "removed_A_elec_field"
    return new


def damage_relax_dynamics_tolerance(a_cat):
    """Slacken the dynamics proxy tolerances - no design ever flags as a
    'static sculpture'. Tests whether E_dynamics_proxy is doing real work."""
    new = copy.deepcopy(a_cat)
    ad = new["channels"].get("A_dynamics")
    if ad:
        ad["max_bb_neighbor_dist_var_A"] = 100.0   # accept anything
        ad["min_secondary_structure_residues_in_active_site"] = 0
    new["_damage"] = "relaxed_A_dynamics_tolerances"
    return new


def damage_null_acat(a_cat):
    new = copy.deepcopy(a_cat)
    for k in ("A_steric", "A_contact", "A_path", "A_anchor", "A_TS", "A_stereo", "A_elec"):
        new["channels"][k] = [] if k != "A_path" else None
    new["_damage"] = "null_acat_all_channels_empty"
    return new


ALL_DAMAGES = {
    "real":                  lambda a: a,
    "rotated_path_90":       lambda a: damage_rotated_path(a, 90),
    "rotated_path_180":      lambda a: damage_rotated_path(a, 180),
    "inverted_face":         damage_inverted_face,
    "wrong_hapticity":       damage_wrong_hapticity,
    "blocked_open_site":     damage_blocked_open_site,
    "rotated_contact_90":    lambda a: damage_rotated_contact(a, 90),
    "shuffled_contact_42":   lambda a: damage_shuffled_contact(a, 42),
    "shuffled_contact_7":    lambda a: damage_shuffled_contact(a, 7),
    "flipped_face":          damage_flip_face,
    "shifted_coord_zones":   lambda a: damage_shift_coord_zones(a, 3.0),
    "flipped_elec_dipole":   damage_flip_elec_dipole,
    "removed_elec_field":    damage_remove_elec_field,
    "relaxed_dynamics":      damage_relax_dynamics_tolerance,
    "null_acat":             damage_null_acat,
}


def generate_all(a_cat):
    return {name: fn(copy.deepcopy(a_cat)) for name, fn in ALL_DAMAGES.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a_cat_json")
    ap.add_argument("--out-dir", default=None,
                    help="dump each damaged A_cat to this dir as <name>.json")
    args = ap.parse_args()
    a = json.load(open(args.a_cat_json))
    variants = generate_all(a)
    print(f"# Generated {len(variants)} variants from {args.a_cat_json}:")
    for name, v in variants.items():
        marker = "(real)" if name == "real" else v.get("_damage", name)
        print(f"  {name:<24}  {marker}")
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            json.dump(v, open(os.path.join(args.out_dir, name + ".json"), "w"), indent=2)


if __name__ == "__main__":
    main()
