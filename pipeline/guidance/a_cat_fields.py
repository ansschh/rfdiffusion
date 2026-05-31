#!/usr/bin/env python3
"""a_cat_fields.py - lift typed-Gaussian A_cat (from instantiate_acat.py) into
continuous fields in the cofactor-local frame.

This is the representation layer for Level-2 catalytic guidance. Each channel
becomes a callable `eval(x_world, y_world, z_world, **kwargs) -> scalar` that
the E_cat terms compose into a catalytic likelihood.

A_cat JSON (channels):
  A_steric    : hard exclusion spheres around cofactor atoms (vdW + 0.4 A)
  A_contact   : typed Gaussians rewarding correct residue-type placement
  A_path      : substrate-approach cone (apex at metal, axis metal->hydride)
  A_anchor    : (diagnostic in v0)
  A_TS        : (low-confidence; not used in E_cat yet)
  A_stereo    : directional bias inside substrate cone (chem mode only)
  A_elec      : charged-residue Gaussian (chem mode only)

Fields exposed:
  eval_path(p_world)                   -> [0,1] occupancy in substrate cone
  eval_contact(p_world, residue_type)  -> [0,inf) reward density
  eval_avoid(p_world)                  -> [0,inf) penalty density
  eval_anchor(p_world, residue_type)   -> [0,inf) anchor-placement reward
  eval_steric(p_world)                 -> [0,inf) hard-exclusion penalty
  eval_elec(p_world, charge_type)      -> [0,inf) electrostatic reward
  eval_stereo(p_world)                 -> [-1,1] face-bias inside cone

All eval methods take WORLD coords (matches PDB) and internally apply the
frame transform world -> local using A_cat.frame.

Author note: these are field FORMS only. Per-term E_cat scorers in
`e_terms/*.py` choose which fields to apply with which weights, evaluated on
protein conformations.
"""
from __future__ import annotations
import json
import math
from typing import Iterable, Optional


def _vsub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def _vdot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def _vlen(v):    return math.sqrt(_vdot(v, v))


def _world_to_local(p_world, origin, R):
    """v_local = R @ (v_world - origin); R rows are local axes in world frame."""
    rel = _vsub(p_world, origin)
    return (R[0][0]*rel[0] + R[0][1]*rel[1] + R[0][2]*rel[2],
            R[1][0]*rel[0] + R[1][1]*rel[1] + R[1][2]*rel[2],
            R[2][0]*rel[0] + R[2][1]*rel[1] + R[2][2]*rel[2])


def _sigmoid(z):
    if z > 30:  return 1.0
    if z < -30: return 0.0
    return 1.0 / (1.0 + math.exp(-z))


class ACatFields:
    """Continuous-field view over an A_cat JSON, evaluable at world coords."""

    def __init__(self, a_cat_dict: dict, *,
                 path_sigma_perp: float = 0.5,
                 path_sigma_par: float = 0.5,
                 path_apex_offset_back: float = 0.0,
                 path_min_extension_past_hydride: float = 0.3,
                 contact_sigma_floor: float = 0.5,
                 steric_softness: float = 0.3,
                 avoid_radius_around_metal: float = 2.0,
                 avoid_sigma: float = 0.5,
                 elec_sigma_default: float = 2.0):
        """Parameters with E_path-relevant defaults visible at top so we can
        show them to the user before locking.

        path_sigma_perp/par: smoothing width on the cone boundary (A). 0.5 A
          gives a soft transition over ~+/-1 A. Smaller -> harder cone, more
          like V_rxn's hard test.
        path_apex_offset_back: shift the geometric cone apex BACK toward the
          metal by this much along the axis. 0.0 -> apex at A_path.apex_local
          (hydride). 1.6 -> apex at the metal (V_rxn convention).
        path_min_extension_past_hydride: do not score atoms closer to apex than
          (|hydride - apex_geom| + this). 0.3 A matches V_rxn's
          `proj <= proj_open + 0.3` skip.
        """
        self.a = a_cat_dict
        self.target = a_cat_dict.get("target")
        self.mode = a_cat_dict.get("mode", "oracle")
        frame = a_cat_dict["frame"]
        self.origin = tuple(frame["origin_world"])
        self.R = frame["R_world_to_local"]
        ch = a_cat_dict.get("channels", {})
        self.A_steric  = ch.get("A_steric", []) or []
        self.A_contact = ch.get("A_contact", []) or []
        self.A_path    = ch.get("A_path")
        self.A_anchor  = ch.get("A_anchor", []) or []
        self.A_stereo  = ch.get("A_stereo")
        a_elec = ch.get("A_elec")
        self.A_elec    = a_elec if isinstance(a_elec, list) else []
        # tunables
        self.path_sigma_perp = path_sigma_perp
        self.path_sigma_par  = path_sigma_par
        self.path_apex_offset_back = path_apex_offset_back
        self.path_min_extension_past_hydride = path_min_extension_past_hydride
        self.contact_sigma_floor = contact_sigma_floor
        self.steric_softness = steric_softness
        self.avoid_radius_around_metal = avoid_radius_around_metal
        self.avoid_sigma = avoid_sigma
        self.elec_sigma_default = elec_sigma_default
        self._prep_path()

    # ---- frame helpers --------------------------------------------------

    def to_local(self, p_world):
        return _world_to_local(p_world, self.origin, self.R)

    # ---- A_path: cone occupancy field ----------------------------------

    def _prep_path(self):
        """Pre-compute the substrate-cone geometry in local frame:
            apex_geom     : actual geometric apex (offset back toward metal)
            axis          : normalized cone axis
            r_par_hydride : parallel projection from apex_geom to hydride
            r_min         : start of scoring region (just past hydride)
            r_max         : end of scoring region (apex_geom + r_par_hydride + extent)
            tan_half      : tangent of half-angle
        """
        self._path_ok = False
        if not self.A_path:
            return
        ap = self.A_path
        apex_local_at_hydride = tuple(ap["apex_local"])
        axis_raw = tuple(ap.get("axis_local", [0.0, 0.0, 1.0]))
        an = _vlen(axis_raw)
        axis = tuple(c / an for c in axis_raw) if an > 0 else (0.0, 0.0, 1.0)
        # geometric apex: shift apex_local_at_hydride BACK along -axis by offset
        d = self.path_apex_offset_back
        apex_geom = (apex_local_at_hydride[0] - axis[0]*d,
                     apex_local_at_hydride[1] - axis[1]*d,
                     apex_local_at_hydride[2] - axis[2]*d)
        # r_par_hydride = how far along axis from apex_geom the hydride sits
        r_par_hydride = (apex_local_at_hydride[0]-apex_geom[0])*axis[0] + \
                        (apex_local_at_hydride[1]-apex_geom[1])*axis[1] + \
                        (apex_local_at_hydride[2]-apex_geom[2])*axis[2]
        half_deg = float(ap.get("half_angle_deg", 30.0))
        extent   = float(ap.get("extent_A", 5.5))
        self._path_apex_geom = apex_geom
        self._path_axis = axis
        self._path_r_par_hydride = r_par_hydride
        self._path_r_min = r_par_hydride + self.path_min_extension_past_hydride
        self._path_r_max = r_par_hydride + extent
        self._path_tan_half = math.tan(math.radians(half_deg))
        self._path_half_deg = half_deg
        self._path_extent = extent
        self._path_ok = True

    def eval_path(self, p_world) -> float:
        """Soft occupancy of the substrate-approach cone at world point p.
        1.0 = deep inside the cone (substrate would live here);
        0.0 = well outside.
        Smoothed by path_sigma_perp on the lateral boundary and path_sigma_par
        on the axial endpoints.
        """
        if not self._path_ok:
            return 0.0
        pl = self.to_local(p_world)
        ap = self._path_apex_geom
        ax = self._path_axis
        rel = (pl[0]-ap[0], pl[1]-ap[1], pl[2]-ap[2])
        r_par = rel[0]*ax[0] + rel[1]*ax[1] + rel[2]*ax[2]
        r_perp = math.sqrt(max(0.0, rel[0]*rel[0]+rel[1]*rel[1]+rel[2]*rel[2] - r_par*r_par))
        # axial gate: r in [r_min, r_max] with soft endpoints
        gate_lo = _sigmoid((r_par - self._path_r_min) / self.path_sigma_par)
        gate_hi = _sigmoid((self._path_r_max - r_par) / self.path_sigma_par)
        # lateral gate: r_perp <= r_par * tan(half), only meaningful inside r_par>0
        cone_r = max(0.0, r_par) * self._path_tan_half
        gate_lat = _sigmoid((cone_r - r_perp) / self.path_sigma_perp)
        return gate_lo * gate_hi * gate_lat

    def path_params(self) -> dict:
        """Inspectable summary of the path field for parameter approval."""
        if not self._path_ok:
            return {"present": False}
        return {
            "present": True,
            "apex_local_input": list(self.A_path["apex_local"]),
            "apex_geom_local_after_back_offset": list(self._path_apex_geom),
            "axis_local": list(self._path_axis),
            "half_angle_deg": self._path_half_deg,
            "extent_A_from_apex_input": self._path_extent,
            "scoring_region_r_par_from_apex_geom": [self._path_r_min, self._path_r_max],
            "tunables": {
                "sigma_perp_A": self.path_sigma_perp,
                "sigma_par_A": self.path_sigma_par,
                "apex_offset_back_A": self.path_apex_offset_back,
                "min_extension_past_hydride_A": self.path_min_extension_past_hydride,
            },
        }

    # ---- A_contact: typed Gaussian field -------------------------------

    def eval_contact(self, p_world, residue_type: str) -> float:
        """Sum of typed Gaussian contributions matching residue_type."""
        if not self.A_contact:
            return 0.0
        pl = self.to_local(p_world)
        s = 0.0
        for g in self.A_contact:
            if g.get("type") != residue_type:
                continue
            mu = g["mu_local"]; sig = g["Sigma_diag"]
            sx = max(self.contact_sigma_floor, sig[0])
            sy = max(self.contact_sigma_floor, sig[1])
            sz = max(self.contact_sigma_floor, sig[2])
            dx = (pl[0]-mu[0])/sx; dy = (pl[1]-mu[1])/sy; dz = (pl[2]-mu[2])/sz
            s += g.get("w", 1.0) * math.exp(-0.5*(dx*dx + dy*dy + dz*dz))
        return s

    def contact_types(self) -> list:
        return sorted({g.get("type", "other") for g in self.A_contact})

    # ---- A_steric: hard exclusion around cofactor atoms ----------------

    def eval_steric(self, p_world) -> float:
        """Penalty density: 1 inside any A_steric sphere, smoothly decaying
        beyond the radius. Sum (not max) across spheres so deep interior of
        the cofactor scores high.
        """
        if not self.A_steric:
            return 0.0
        pl = self.to_local(p_world)
        s = 0.0
        for sph in self.A_steric:
            mu = sph["pos_local"]; r = sph["r"]
            dx = pl[0]-mu[0]; dy = pl[1]-mu[1]; dz = pl[2]-mu[2]
            d = math.sqrt(dx*dx + dy*dy + dz*dz)
            s += _sigmoid((r - d) / self.steric_softness)
        return s

    # ---- A_avoid: derived no-go zones (no explicit channel yet) --------

    def eval_avoid(self, p_world) -> float:
        """Avoid regions (v0 derived rule): a sphere just outside the metal at
        radius `avoid_radius_around_metal` discourages donor placement that
        would coordinate the metal from a non-allowed direction (i.e., outside
        any A_contact region; donor poisoning).

        v0 is intentionally minimal: just the donor-poisoning shell around the
        metal. Future versions can derive avoid regions from V_chem forbidden
        coordination geometries and from substrate-cone-adjacent regions.
        """
        pl = self.to_local(p_world)
        d_metal = math.sqrt(pl[0]*pl[0] + pl[1]*pl[1] + pl[2]*pl[2])
        # bump centered at avoid_radius_around_metal, width avoid_sigma
        z = (d_metal - self.avoid_radius_around_metal) / self.avoid_sigma
        return math.exp(-0.5 * z * z)

    # ---- A_anchor: anchor-residue placement (diagnostic in v0) ---------

    def eval_anchor(self, p_world, residue_type: str) -> float:
        """v0: returns 0 unless A_anchor has carried entries (currently
        diagnostic-only per manifest.anchor.treat_as)."""
        return 0.0

    # ---- A_elec: charged-residue Gaussian (chem mode) ------------------

    def eval_elec(self, p_world, charge_type: str) -> float:
        if not self.A_elec:
            return 0.0
        pl = self.to_local(p_world)
        s = 0.0
        for g in self.A_elec:
            if g.get("type") != charge_type:
                continue
            mu = g["mu_local"]; sig = g.get("Sigma_diag", [self.elec_sigma_default]*3)
            sx = max(self.contact_sigma_floor, sig[0])
            sy = max(self.contact_sigma_floor, sig[1])
            sz = max(self.contact_sigma_floor, sig[2])
            dx = (pl[0]-mu[0])/sx; dy = (pl[1]-mu[1])/sy; dz = (pl[2]-mu[2])/sz
            s += g.get("w", 1.0) * math.exp(-0.5*(dx*dx + dy*dy + dz*dz))
        return s

    # ---- A_stereo: face bias inside cone -------------------------------

    def eval_stereo(self, p_world) -> float:
        """Returns a signed value in roughly [-1,1] for the side of the
        substrate-cone where the chiral N,N face prefers contact. Zero if no
        A_stereo (oracle mode or symmetric cofactor)."""
        if not self.A_stereo or not self._path_ok:
            return 0.0
        pl = self.to_local(p_world)
        v = self.A_stereo["v_stereo_local"]
        # only meaningful inside cone
        occ = self.eval_path(p_world)
        if occ < 0.05:
            return 0.0
        # project local position onto v_stereo (relative to cone axis)
        return self.A_stereo.get("bias_strength", 0.3) * occ * (
            pl[0]*v[0] + pl[1]*v[1] + pl[2]*v[2]
        )


# ---- loader ----------------------------------------------------------------

def load(a_cat_path: str, **tunables) -> ACatFields:
    return ACatFields(json.load(open(a_cat_path)), **tunables)


if __name__ == "__main__":
    import argparse, pprint, sys
    ap = argparse.ArgumentParser(description="Inspect ACatFields for a target")
    ap.add_argument("a_cat_json")
    ap.add_argument("--probe", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="probe point in WORLD coords; prints all field values")
    args = ap.parse_args()
    f = load(args.a_cat_json)
    print(f"Target: {f.target}  mode={f.mode}")
    print(f"Frame origin (world): {f.origin}")
    print(f"R_world_to_local:")
    for row in f.R: print(f"   {row}")
    print(f"\nA_path summary:")
    pprint.pprint(f.path_params(), sort_dicts=False)
    print(f"\nA_contact types: {f.contact_types()}  (n={len(f.A_contact)} Gaussians)")
    print(f"A_steric spheres: {len(f.A_steric)}")
    print(f"A_elec: {len(f.A_elec)} Gaussians")
    print(f"A_stereo: {'yes' if f.A_stereo else 'no'}")
    if args.probe:
        p = tuple(args.probe)
        pl = f.to_local(p)
        print(f"\nProbe world={p}  local={tuple(round(c,3) for c in pl)}")
        print(f"  eval_path     = {f.eval_path(p):.4f}")
        for t in f.contact_types():
            print(f"  eval_contact[{t:>14}] = {f.eval_contact(p, t):.4f}")
        print(f"  eval_steric   = {f.eval_steric(p):.4f}")
        print(f"  eval_avoid    = {f.eval_avoid(p):.4f}")
        for t in {g.get('type') for g in f.A_elec}:
            print(f"  eval_elec[{t}] = {f.eval_elec(p, t):.4f}")
        if f.A_stereo:
            print(f"  eval_stereo   = {f.eval_stereo(p):.4f}")
