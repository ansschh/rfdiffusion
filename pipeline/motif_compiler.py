#!/usr/bin/env python3
"""
motif_compiler.py — compile a validated ArM trajectory into an RFD2 active-site input.

Bridge for Workstream-1 step 3 (Sub-question B): turn a curated, CSD-verified
trajectory X into exactly the inputs RFD2's run_inference.py consumes — the motif
PDB (cofactor reactive core), the contig string, the ligand spec, and the
partially-fixed-ligand atom list — matching the format proven by the open_source_demo
generation smoke test.

Cofactor representation strategy = 'reactive_core_cp_body' (user-chosen):
  Fix the catalytically essential atoms as the RFD2 ligand —
    metal + retained kappa-N,N chelate donors + the active-species hydride
    + the eta5 Cp* ring carbons kept as a RIGID STERIC BODY (so the chiral pocket
    the protein must accommodate is preserved) —
  but do NOT try to encode eta5 pi-bonding that RFD2/RFAA never learned, and DROP the
  labile leg (Cl). Substrate/TS placement is a separate, explicitly-flagged step.

Design rules honored:
  * No regex/text-mining of the nuanced extraction. The chemistry was read by the
    extraction agents into reconciled.md; this tool reads only structured JSON
    (motif_targets.json) + real PDB geometry. Atom selection is by deterministic
    interatomic distance, gated by CSD bond-length p99 cutoffs from the target's
    deterministic_checks.json. Code does numeric/deterministic work only.

Usage:
  python motif_compiler.py 3ZP9 \
      [--targets pipeline/motif_targets.json] \
      [--audit audit/subQB_curation/extractions] \
      [--pdb-dir <dir with <id>.pdb, else fetched from RCSB>] \
      [--out pipeline/compiled] \
      [--scaffold-length 160] [--num-designs 100]
"""
from __future__ import annotations
import argparse, json, math, os, sys, urllib.request
from dataclasses import dataclass, field

# Generous element-pair fallback cutoffs (Angstrom) for metal--donor bonds, used only
# when the target's deterministic_checks.json has no CSD p99 for that pair.
FALLBACK_CUTOFF = {"C": 2.40, "N": 2.55, "O": 2.65, "S": 2.85, "CL": 2.90, "P": 2.85}


@dataclass
class Atom:
    serial: int
    name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    x: float
    y: float
    z: float
    occ: float
    element: str

    @property
    def xyz(self):
        return (self.x, self.y, self.z)


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def parse_pdb(path):
    """Deterministic fixed-column PDB parse of ATOM/HETATM records."""
    atoms = []
    with open(path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                element = line[76:78].strip().upper()
                name = line[12:16].strip()
                if not element:  # fall back to atom name's leading alpha
                    element = "".join(c for c in name if c.isalpha())[:2].upper()
                atoms.append(Atom(
                    serial=int(line[6:11]),
                    name=name,
                    altloc=line[16].strip(),
                    resname=line[17:20].strip(),
                    chain=line[21].strip(),
                    resseq=int(line[22:26]),
                    x=float(line[30:38]), y=float(line[38:46]), z=float(line[46:54]),
                    occ=float(line[54:60] or "1.0"),
                    element=element,
                ))
            except ValueError:
                continue
    return atoms


def fetch_pdb(pdb_id, dest):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"[fetch] {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def load_csd_cutoffs(audit_dir, pdb_id, metal_el):
    """Read CSD bond-length p99 per donor element from deterministic_checks.json."""
    cutoffs = {}
    p = os.path.join(audit_dir, pdb_id, "deterministic_checks.json")
    if not os.path.isfile(p):
        return cutoffs
    data = json.load(open(p))
    dists = data.get("checks", {}).get("bond_distributions_CSD", {})
    for label, stats in dists.items():
        # labels look like "Ir-C (eta5-Cp*)", "Ir-N (pyridyl-N)", "Ir-Cl (chloride)"
        u = label.upper()
        for el in FALLBACK_CUTOFF:
            token = f"-{el}" if el != "CL" else "-CL"
            if token in u.replace(" ", "") or f"{metal_el}-{el}".upper() in u.replace(" ", ""):
                p99 = stats.get("p99")
                if p99 is not None:
                    # keep the LARGEST p99 seen for that element (most permissive)
                    cutoffs[el] = max(cutoffs.get(el, 0.0), float(p99))
    return cutoffs


def cutoff_for(element, csd_cutoffs):
    return csd_cutoffs.get(element, FALLBACK_CUTOFF.get(element, 2.6))


def find_metal(atoms, resname, metal_el):
    cands = [a for a in atoms if a.resname == resname and a.element == metal_el]
    if not cands:
        cands = [a for a in atoms if a.element == metal_el]
    if not cands:
        raise SystemExit(f"metal {metal_el} not found in {resname}")
    # highest-occupancy copy (handles alt-loc / partial occupancy cofactors)
    return max(cands, key=lambda a: a.occ)


def first_sphere(atoms, metal, csd_cutoffs, max_radius=3.2):
    """Atoms within element-specific bond cutoff of the metal, alt-loc consistent."""
    keep_alt = {"", metal.altloc} if metal.altloc else {""}
    donors = []
    for a in atoms:
        if a.serial == metal.serial:
            continue
        if a.altloc and a.altloc not in keep_alt:
            continue
        d = dist(metal.xyz, a.xyz)
        if d <= max_radius and d <= cutoff_for(a.element, csd_cutoffs):
            donors.append((a, d))
    donors.sort(key=lambda t: t[1])
    return donors


def normalize(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v) if n else v


def synth_hydride(metal, leg_atom, ir_h):
    """Place a hydride along the metal->labile-leg vector at ir_h Angstrom."""
    v = normalize((leg_atom.x - metal.x, leg_atom.y - metal.y, leg_atom.z - metal.z))
    return (metal.x + v[0] * ir_h, metal.y + v[1] * ir_h, metal.z + v[2] * ir_h)


def select_reactive_core(atoms, spec, metal, donors):
    """Strategy 'reactive_core_cp_body'. Returns (core_atoms[list[Atom]], report dict)."""
    fs = spec["first_sphere"]
    retained = set(fs.get("retained_donors", []))
    labile = fs.get("labile_leg")
    metal_el = spec["catalytic_metal"]["element"]

    donor_by_name = {a.name: (a, d) for a, d in donors}
    report = {"metal": {"name": metal.name, "occ": metal.occ, "xyz": metal.xyz},
              "donors_detected": [{"name": a.name, "element": a.element, "dist": round(d, 3)}
                                  for a, d in donors],
              "selected": [], "synthesized": [], "dropped": [], "warnings": []}

    core = [metal]
    report["selected"].append(f"{metal.name} (metal {metal_el})")

    # retained chelate donors (must be in first sphere)
    for nm in retained:
        if nm in donor_by_name:
            a, d = donor_by_name[nm]
            core.append(a)
            report["selected"].append(f"{nm} ({a.element}, {d:.2f} A, retained donor)")
        else:
            report["warnings"].append(f"retained donor {nm} NOT within bond cutoff of metal")

    # Cp* ring: the 5 closest carbons in the eta5 band that aren't already chosen
    cstar_band = [(a, d) for a, d in donors if a.element == "C" and a.name not in retained]
    cstar = cstar_band[:5]
    if len(cstar) < 5:
        report["warnings"].append(f"only {len(cstar)} eta5 Cp* carbons within cutoff (expected 5)")
    for a, d in cstar:
        core.append(a)
        report["selected"].append(f"{a.name} (C, {d:.2f} A, Cp* ring -- rigid steric body)")

    # drop the labile leg
    if labile and labile in donor_by_name:
        a, d = donor_by_name[labile]
        report["dropped"].append(f"{labile} ({a.element}, {d:.2f} A, labile leg -> replaced by hydride)")

    # synthesize the active-species hydride along metal->labile vector
    act = spec.get("active_species", {}).get("hydride")
    if act and act.get("synthesize") and labile in donor_by_name:
        leg, _ = donor_by_name[labile]
        hx = synth_hydride(metal, leg, float(act.get("ir_h_distance", 1.6)))
        h = Atom(serial=99999, name=act.get("name", "H1"), altloc="", resname="LIG",
                 chain="L", resseq=1, x=hx[0], y=hx[1], z=hx[2], occ=1.0, element="H")
        core.append(h)
        report["synthesized"].append(
            f"{h.name} (H, hydride at {act.get('ir_h_distance',1.6)} A along {metal.name}->{labile}; "
            f"{act.get('confidence','inferred')})")
    return core, report


def write_motif_pdb(core, out_path, lig_resname="LIG", chain="L"):
    """Write the reactive core as a single ligand residue of HETATM records."""
    lines = []
    for i, a in enumerate(core, start=1):
        nm = a.name
        # PDB atom-name column convention: pad so element aligns
        name_field = (f"{nm:<3}" if len(nm) >= 4 else f" {nm:<3}")[:4]
        lines.append(
            f"HETATM{i:>5} {name_field}{lig_resname:>4} {chain}{1:>4}    "
            f"{a.x:8.3f}{a.y:8.3f}{a.z:8.3f}{1.00:6.2f}{0.00:6.2f}          {a.element:>2}"
        )
    lines.append("END")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return out_path


def build_rfd2_command(spec, core, motif_pdb, lig_resname, out_prefix,
                       scaffold_length, num_designs):
    """Emit the run_inference.py invocation for ligand-conditioned pocket generation."""
    core_atom_names = [a.name for a in core]
    fixed = ",".join(core_atom_names)
    contig = f"['{scaffold_length}-{scaffold_length}']"
    cmd = (
        "rf_diffusion/run_inference.py --config-name=aa "
        "inference.deterministic=True "
        "inference.ckpt_path=REPO_ROOT/rf_diffusion/model_weights/RFD_173.pt "
        f"inference.input_pdb={motif_pdb} "
        f"inference.ligand={lig_resname} "
        f"contigmap.contigs={contig} "
        f"++inference.partially_fixed_ligand=\"{{{lig_resname}:[{fixed}]}}\" "
        f"inference.num_designs={num_designs} inference.design_startnum=0 "
        f"inference.output_prefix={out_prefix} "
        "hydra.job_logging.root.level=WARN"
    )
    return cmd, core_atom_names


def compile_target(pdb_id, targets, audit_dir, pdb_dir, out_root,
                   scaffold_length=None, num_designs=None):
    spec = targets["targets"][pdb_id]
    metal_el = spec["catalytic_metal"]["element"]
    resname = spec["cofactor_resname"]
    scaffold_length = scaffold_length or spec["design"]["scaffold_length"]
    num_designs = num_designs or spec["design"].get("num_designs_default", 100)

    pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    if not os.path.isfile(pdb_path):
        os.makedirs(pdb_dir, exist_ok=True)
        fetch_pdb(pdb_id, pdb_path)

    atoms = parse_pdb(pdb_path)
    csd_cutoffs = load_csd_cutoffs(audit_dir, pdb_id, metal_el)
    metal = find_metal(atoms, resname, metal_el)
    donors = first_sphere(atoms, metal, csd_cutoffs)
    core, report = select_reactive_core(atoms, spec, metal, donors)

    out_dir = os.path.join(out_root, pdb_id)
    os.makedirs(out_dir, exist_ok=True)
    motif_pdb = os.path.join(out_dir, "motif.pdb")
    write_motif_pdb(core, motif_pdb, lig_resname="LIG")

    out_prefix = f"$SCRATCH/RFdiffusion2/arm_designs/{pdb_id}/{pdb_id}_cond0"
    cmd, core_names = build_rfd2_command(
        spec, core, f"$ARM/{pdb_id}/motif.pdb", "LIG", out_prefix,
        scaffold_length, num_designs)

    manifest = {
        "pdb_id": pdb_id,
        "strategy": "reactive_core_cp_body",
        "csd_cutoffs_used": csd_cutoffs or "fallback (no deterministic_checks.json found)",
        "scaffold_length": scaffold_length,
        "num_designs": num_designs,
        "core_atoms": core_names,
        "selection_report": report,
        "anchor": {**spec.get("anchor", {}), "fixed_in_motif": False},
        "substrate": {**spec.get("substrate", {}),
                      "placed_in_v1": spec.get("substrate", {}).get("place_in_v1", False)},
        "rfd2_command": cmd,
        "verification": spec.get("verification"),
        "provenance": spec.get("provenance"),
        "notes": [
            "Cp* ring kept as a rigid steric body; eta5 pi-bonding NOT encoded (RFD2 has no eta5 term).",
            "Labile leg dropped; hydride synthesized (NOT crystallographic) -> active-species core.",
            "Anchor (dative-to-CA-II-Zn) treated as a diagnostic, not a fixed motif element.",
            "Substrate/TS pose deferred: 3ZP9 has no bound substrate and g-double-dagger is transferred (LOW conf).",
            "$ARM = the dir holding compiled/<id>/; set it on the cluster before running the command.",
        ],
    }
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    with open(os.path.join(out_dir, "run_inference.cmd"), "w") as fh:
        fh.write(cmd + "\n")
    return out_dir, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb_id")
    ap.add_argument("--targets", default="pipeline/motif_targets.json")
    ap.add_argument("--audit", default="audit/subQB_curation/extractions")
    ap.add_argument("--pdb-dir", default="pipeline/pdb")
    ap.add_argument("--out", default="pipeline/compiled")
    ap.add_argument("--scaffold-length", type=int, default=None)
    ap.add_argument("--num-designs", type=int, default=None)
    args = ap.parse_args()

    targets = json.load(open(args.targets))
    if args.pdb_id not in targets["targets"]:
        raise SystemExit(f"{args.pdb_id} not in {args.targets}")
    out_dir, manifest = compile_target(
        args.pdb_id, targets, args.audit, args.pdb_dir, args.out,
        args.scaffold_length, args.num_designs)

    r = manifest["selection_report"]
    print(f"=== motif compiled: {args.pdb_id} -> {out_dir} ===")
    print(f"metal: {r['metal']['name']} (occ {r['metal']['occ']})")
    print("first-sphere donors detected (name, element, dist A):")
    for d in r["donors_detected"]:
        print(f"   {d['name']:>4} {d['element']:>2}  {d['dist']}")
    print("SELECTED into reactive core:")
    for s in r["selected"]:
        print(f"   + {s}")
    for s in r["synthesized"]:
        print(f"   ~ {s}")
    for s in r["dropped"]:
        print(f"   - {s}")
    for w in r["warnings"]:
        print(f"   ! {w}")
    print(f"\ncore atoms ({len(manifest['core_atoms'])}): {manifest['core_atoms']}")
    print(f"\nRFD2 command:\n{manifest['rfd2_command']}")


if __name__ == "__main__":
    main()
