#!/usr/bin/env python3
"""
motif_compiler.py — compile a validated ArM trajectory into an RFD2 active-site input.

Bridge for Workstream-1 step 3 (Sub-question B): turn a curated, CSD-verified
trajectory X into exactly the inputs RFD2's run_inference.py consumes — the motif
PDB (cofactor reactive core + guidepost residues), the contig, the ligand spec, and
contig_atoms — matching the format proven by the open_source_demo generation smoke.

WHY GUIDEPOSTS: this RFD2 build has no ligand-only path — make_indep builds idx_polymer
from protein residues in the input PDB and crashes if there are none (shipped "binder"
inputs carry a trimmed pocket; "heme" carries the whole scaffold). The RFD2-native mode
is active-site scaffolding: fix a few residues + the ligand, generate the rest. For an
ArM (the metal is the catalyst, not a sidechain) the honest guideposts are the cofactor's
nearest second-sphere contacts, selected BY DISTANCE from the validated affordance map.
RFD2 still builds a fresh fold; this is not pocket reconstruction (pocket-RMSD = diagnostic).

Cofactor representation strategy = 'reactive_core_cp_body' (user-chosen):
  metal + retained kappa-N,N chelate donors + synthesized active-species hydride
  + eta5 Cp* ring carbons kept as a RIGID STERIC BODY; eta5 pi-bonding NOT encoded;
  labile leg (Cl) dropped. Substrate/TS placement is a separate, flagged step.

Design rules honored: no regex/text-mining of the nuanced extraction (chemistry was read
by the extraction agents into reconciled.md; here we read only structured JSON + real PDB
geometry; all atom/residue selection is by deterministic interatomic distance).

Usage:
  python motif_compiler.py 3ZP9 [--targets ...] [--audit ...] [--pdb-dir ...] [--out ...]
                                [--scaffold-length 160] [--num-designs 100] [--guideposts N]
"""
from __future__ import annotations
import argparse, json, math, os, urllib.request
from dataclasses import dataclass

FALLBACK_CUTOFF = {"C": 2.40, "N": 2.55, "O": 2.65, "S": 2.85, "CL": 2.90, "P": 2.85}
BACKBONE = {"N", "CA", "C", "O", "OXT"}


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
    record: str = "ATOM"

    @property
    def xyz(self):
        return (self.x, self.y, self.z)


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def parse_pdb(path):
    atoms = []
    with open(path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            try:
                element = line[76:78].strip().upper()
                name = line[12:16].strip()
                if not element:
                    element = "".join(c for c in name if c.isalpha())[:2].upper()
                atoms.append(Atom(
                    serial=int(line[6:11]), name=name, altloc=line[16].strip(),
                    resname=line[17:20].strip(), chain=line[21].strip(),
                    resseq=int(line[22:26]),
                    x=float(line[30:38]), y=float(line[38:46]), z=float(line[46:54]),
                    occ=float(line[54:60] or "1.0"), element=element, record=rec,
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
    cutoffs = {}
    p = os.path.join(audit_dir, pdb_id, "deterministic_checks.json")
    if not os.path.isfile(p):
        return cutoffs
    dists = json.load(open(p)).get("checks", {}).get("bond_distributions_CSD", {})
    for label, stats in dists.items():
        u = label.upper().replace(" ", "")
        for el in FALLBACK_CUTOFF:
            if f"-{el}" in u and stats.get("p99") is not None:
                cutoffs[el] = max(cutoffs.get(el, 0.0), float(stats["p99"]))
    return cutoffs


def cutoff_for(el, csd):
    return csd.get(el, FALLBACK_CUTOFF.get(el, 2.6))


def find_metal(atoms, resname, metal_el):
    cands = [a for a in atoms if a.resname == resname and a.element == metal_el] \
        or [a for a in atoms if a.element == metal_el]
    if not cands:
        raise SystemExit(f"metal {metal_el} not found in {resname}")
    return max(cands, key=lambda a: a.occ)


def first_sphere(atoms, metal, csd, max_radius=3.2):
    keep = {"", metal.altloc} if metal.altloc else {""}
    out = []
    for a in atoms:
        if a.serial == metal.serial or (a.altloc and a.altloc not in keep):
            continue
        d = dist(metal.xyz, a.xyz)
        if d <= max_radius and d <= cutoff_for(a.element, csd):
            out.append((a, d))
    out.sort(key=lambda t: t[1])
    return out


def normalize(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v) if n else v


def synth_hydride(metal, leg, ir_h):
    v = normalize((leg.x - metal.x, leg.y - metal.y, leg.z - metal.z))
    return (metal.x + v[0] * ir_h, metal.y + v[1] * ir_h, metal.z + v[2] * ir_h)


def select_reactive_core(atoms, spec, metal, donors):
    fs = spec["first_sphere"]
    retained = set(fs.get("retained_donors", []))
    labile = fs.get("labile_leg")
    by_name = {a.name: (a, d) for a, d in donors}
    report = {"metal": {"name": metal.name, "occ": metal.occ},
              "donors_detected": [{"name": a.name, "element": a.element, "dist": round(d, 3)}
                                  for a, d in donors],
              "selected": [], "synthesized": [], "dropped": [], "warnings": []}
    core = [metal]
    report["selected"].append(f"{metal.name} (metal {metal.element})")
    for nm in retained:
        if nm in by_name:
            a, d = by_name[nm]; core.append(a)
            report["selected"].append(f"{nm} ({a.element}, {d:.2f} A, retained donor)")
        else:
            report["warnings"].append(f"retained donor {nm} not within bond cutoff")
    cstar = [(a, d) for a, d in donors if a.element == "C" and a.name not in retained][:5]
    if len(cstar) < 5:
        report["warnings"].append(f"only {len(cstar)} Cp* carbons within cutoff (expected 5)")
    for a, d in cstar:
        core.append(a)
        report["selected"].append(f"{a.name} (C, {d:.2f} A, Cp* ring -- rigid steric body)")
    if labile and labile in by_name:
        a, d = by_name[labile]
        report["dropped"].append(f"{labile} ({a.element}, {d:.2f} A, labile leg -> hydride)")
    act = spec.get("active_species", {}).get("hydride")
    if act and act.get("synthesize") and labile in by_name:
        leg = by_name[labile][0]
        hx = synth_hydride(metal, leg, float(act.get("ir_h_distance", 1.6)))
        core.append(Atom(99999, act.get("name", "H1"), "", "LIG", "L", 1,
                         hx[0], hx[1], hx[2], 1.0, "H", "HETATM"))
        report["synthesized"].append(
            f"{act.get('name','H1')} (H, hydride {act.get('ir_h_distance',1.6)} A along "
            f"{metal.name}->{labile}; {act.get('confidence','inferred')})")
    return core, report


def residue_atoms(atoms, chain, resseq, keep_alt):
    return [a for a in atoms if a.chain == chain and a.resseq == resseq
            and (not a.altloc or a.altloc in keep_alt)]


def select_guideposts(atoms, core, spec, metal, n):
    """Pick the n candidate second-sphere residues whose sidechains come closest to the
    reactive core, by deterministic distance. Anchor atoms = 3 sidechain heavies nearest
    the core."""
    keep = {"", metal.altloc} if metal.altloc else {""}
    cands = spec.get("guideposts", {}).get("candidates", [])
    core_xyz = [a.xyz for a in core]
    scored = []
    for c in cands:
        res = residue_atoms(atoms, c["chain"], c["resseq"], keep)
        if not res:
            continue
        heavies = [a for a in res if a.element != "H" and a.name not in BACKBONE]
        if not heavies:
            continue
        # min distance from any sidechain heavy to any core atom
        pairs = [(min(dist(a.xyz, cx) for cx in core_xyz), a) for a in heavies]
        mind = min(p[0] for p in pairs)
        anchors = [a.name for _, a in sorted(pairs, key=lambda p: p[0])[:3]]
        scored.append({"chain": c["chain"], "resseq": c["resseq"],
                       "resname": res[0].resname, "role": c.get("role", ""),
                       "min_dist_to_core": round(mind, 2), "anchor_atoms": anchors,
                       "atoms": res})
    scored.sort(key=lambda g: g["min_dist_to_core"])
    return scored[:n]


def build_contig(guideposts, total_len):
    """Interleave generated gaps with the (resseq-sorted) guidepost residues; gaps sum to
    total_len - len(guideposts). contig_as_guidepost=True => positions are unindexed."""
    gps = sorted(guideposts, key=lambda g: g["resseq"])
    n = len(gps)
    gen = max(total_len - n, n + 1)
    segs = n + 1
    base, rem = divmod(gen, segs)
    gaps = [base + (1 if i < rem else 0) for i in range(segs)]
    parts = []
    for i, g in enumerate(gps):
        parts.append(str(gaps[i]))
        parts.append(f"{g['chain']}{g['resseq']}-{g['resseq']}")
    parts.append(str(gaps[-1]))
    return "['" + ",".join(parts) + "']"


def build_contig_atoms(guideposts):
    items = [f"'{g['chain']}{g['resseq']}':'{','.join(g['anchor_atoms'])}'"
             for g in sorted(guideposts, key=lambda g: g["resseq"])]
    return "{" + ",".join(items) + "}"


def fmt_atom(a, serial, resname=None, chain=None, resseq=None, record=None):
    nm = a.name
    name_field = (f"{nm:<3}" if len(nm) >= 4 else f" {nm:<3}")[:4]
    return (f"{(record or a.record):<6}{serial:>5} {name_field}{(resname or a.resname):>4} "
            f"{(chain or a.chain)}{(resseq if resseq is not None else a.resseq):>4}    "
            f"{a.x:8.3f}{a.y:8.3f}{a.z:8.3f}{1.00:6.2f}{0.00:6.2f}          {a.element:>2}")


def write_motif_pdb(core, guideposts, out_path, lig_resname="LIG", lig_chain="L"):
    """Guidepost protein residues (real coords) + the rigid cofactor core (ligand)."""
    lines, serial = [], 0
    for g in sorted(guideposts, key=lambda g: g["resseq"]):
        for a in g["atoms"]:
            serial += 1
            lines.append(fmt_atom(a, serial, record="ATOM"))
    lines.append("TER")
    for a in core:
        serial += 1
        lines.append(fmt_atom(a, serial, resname=lig_resname, chain=lig_chain,
                              resseq=1, record="HETATM"))
    lines.append("END")
    open(out_path, "w").write("\n".join(lines) + "\n")
    return out_path


def build_rfd2_command(motif_pdb, lig_resname, contig, contig_atoms, out_prefix, num_designs):
    return (
        "rf_diffusion/run_inference.py --config-name=aa "
        "inference.deterministic=True "
        "inference.ckpt_path=REPO_ROOT/rf_diffusion/model_weights/RFD_173.pt "
        f"inference.input_pdb={motif_pdb} inference.ligand={lig_resname} "
        f"contigmap.contigs=\"{contig}\" inference.contig_as_guidepost=True "
        f"contigmap.contig_atoms=\"{contig_atoms}\" "
        f"inference.num_designs={num_designs} inference.design_startnum=0 "
        f"inference.output_prefix={out_prefix} hydra.job_logging.root.level=WARN"
    )


def compile_target(pdb_id, targets, audit_dir, pdb_dir, out_root,
                   scaffold_length=None, num_designs=None, num_guideposts=None):
    spec = targets["targets"][pdb_id]
    metal_el, resname = spec["catalytic_metal"]["element"], spec["cofactor_resname"]
    scaffold_length = scaffold_length or spec["design"]["scaffold_length"]
    num_designs = num_designs or spec["design"].get("num_designs_default", 100)
    num_guideposts = num_guideposts or spec.get("guideposts", {}).get("num_guideposts", 4)

    pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    if not os.path.isfile(pdb_path):
        os.makedirs(pdb_dir, exist_ok=True)
        fetch_pdb(pdb_id, pdb_path)

    atoms = parse_pdb(pdb_path)
    csd = load_csd_cutoffs(audit_dir, pdb_id, metal_el)
    metal = find_metal(atoms, resname, metal_el)
    donors = first_sphere(atoms, metal, csd)
    core, report = select_reactive_core(atoms, spec, metal, donors)
    guideposts = select_guideposts(atoms, core, spec, metal, num_guideposts)
    if not guideposts:
        raise SystemExit("no guidepost residues resolved — check guideposts.candidates")

    contig = build_contig(guideposts, scaffold_length)
    contig_atoms = build_contig_atoms(guideposts)

    out_dir = os.path.join(out_root, pdb_id)
    os.makedirs(out_dir, exist_ok=True)
    write_motif_pdb(core, guideposts, os.path.join(out_dir, "motif.pdb"))

    out_prefix = f"$SCRATCH/RFdiffusion2/arm_designs/{pdb_id}/{pdb_id}_cond0"
    cmd = build_rfd2_command(f"$ARM/{pdb_id}/motif.pdb", "LIG", contig, contig_atoms,
                             out_prefix, num_designs)

    manifest = {
        "pdb_id": pdb_id, "strategy": "reactive_core_cp_body + nearest-contact guideposts",
        "scaffold_length": scaffold_length, "num_designs": num_designs,
        "num_guideposts": len(guideposts),
        "core_atoms": [a.name for a in core],
        "guideposts": [{k: g[k] for k in ("chain", "resseq", "resname", "role",
                                          "min_dist_to_core", "anchor_atoms")} for g in guideposts],
        "contig": contig, "contig_atoms": contig_atoms, "ligand": "LIG",
        "selection_report": report,
        "anchor": {**spec.get("anchor", {}), "fixed_in_motif": False},
        "substrate": {**spec.get("substrate", {})},
        "rfd2_command": cmd,
        "verification": spec.get("verification"), "provenance": spec.get("provenance"),
        "notes": [
            "Active-site scaffolding (RFD2-native): guideposts + rigid cofactor ligand -> fresh fold.",
            "Guideposts = nearest cofactor-contacting 2nd-sphere residues (by distance); NOT pocket reconstruction.",
            "Cp* ring kept as rigid steric body; eta5 NOT encoded; labile Cl dropped; hydride synthesized (not crystallographic).",
            "Anchor (dative-to-CA-II-Zn) = diagnostic, not fixed. Substrate/TS pose deferred (3ZP9 has no bound substrate; g-dd transferred, LOW conf).",
            "$ARM = dir holding compiled/<id>/ on the cluster.",
        ],
    }
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    open(os.path.join(out_dir, "run_inference.cmd"), "w").write(cmd + "\n")
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
    ap.add_argument("--guideposts", type=int, default=None)
    a = ap.parse_args()
    targets = json.load(open(a.targets))
    if a.pdb_id not in targets["targets"]:
        raise SystemExit(f"{a.pdb_id} not in {a.targets}")
    out_dir, m = compile_target(a.pdb_id, targets, a.audit, a.pdb_dir, a.out,
                                a.scaffold_length, a.num_designs, a.guideposts)
    r = m["selection_report"]
    print(f"=== motif compiled: {a.pdb_id} -> {out_dir} ===")
    print(f"metal {r['metal']['name']} (occ {r['metal']['occ']})")
    print("reactive core:")
    for s in r["selected"]:    print(f"   + {s}")
    for s in r["synthesized"]: print(f"   ~ {s}")
    for s in r["dropped"]:     print(f"   - {s}")
    for w in r["warnings"]:    print(f"   ! {w}")
    print(f"core atoms ({len(m['core_atoms'])}): {m['core_atoms']}")
    print("guideposts (nearest cofactor contacts):")
    for g in m["guideposts"]:
        print(f"   {g['resname']}{g['resseq']} {g['chain']}  min {g['min_dist_to_core']} A  "
              f"anchor {g['anchor_atoms']}  [{g['role']}]")
    print(f"contig: {m['contig']}")
    print(f"contig_atoms: {m['contig_atoms']}")
    print(f"\nRFD2 command:\n{m['rfd2_command']}")


if __name__ == "__main__":
    main()
