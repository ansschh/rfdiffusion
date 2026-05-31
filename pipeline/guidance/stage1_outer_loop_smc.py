#!/usr/bin/env python3
"""stage1_outer_loop_smc.py - outer-loop SMC for organometallic ArM design.

This is Stage 1 of the Level-2 plan. RFD2 internals are unchanged. SMC acts
between RFD2 batches:

  particles: K guidepost configurations theta_k (jittered from a base motif)
  weights:   w_k proportional to exp(-lambda * E_cat_k_summary)
  resample:  theta_{wave+1} sampled from {theta_wave^(k)} with probabilities w_k
             + diversification jitter (gaussian on each guidepost position)

Per wave:
  1. Apply jitter to base motif to produce K particle motifs.
  2. For each particle: submit RFD2 to generate N designs.
  3. Score each design with E_cat. Per-particle summary E_cat (min or mean).
  4. Compute weights w_k.
  5. Resample K particles from current with replacement, weighted by w_k.
  6. Add diversification jitter to next-wave particles.
  7. Repeat for n_waves.

The orchestrator emits SLURM sbatch scripts; can also run DRY (no actual
submission) for code-validation.

CLI:
  python stage1_outer_loop_smc.py init --target 3ZP9 --k 8 --n 50 --waves 3 \
      --base-motif pipeline/compiled/3ZP9/motif.pdb \
      --workdir runs/smc/3ZP9 [--dry-run]

  python stage1_outer_loop_smc.py score-wave --workdir runs/smc/3ZP9 --wave 0
  python stage1_outer_loop_smc.py resample --workdir runs/smc/3ZP9 --wave 0
"""
from __future__ import annotations
import argparse, json, math, os, random, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guidance.a_cat_fields import ACatFields
from guidance.e_cat import e_cat
from guidance.jitter_motif import jitter_pdb


def workdir_path(workdir, *parts):
    p = os.path.join(workdir, *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p


def particle_dir(workdir, wave, k):
    """Return a posix-style path so sbatch files work on Linux clusters even
    when generated on Windows."""
    p = workdir_path(workdir, f"wave_{wave:03d}", f"particle_{k:03d}")
    return p.replace("\\", "/")


def init_run(args):
    """Set up the workdir, write the base config, generate wave-0 particles."""
    wd = args.workdir
    os.makedirs(wd, exist_ok=True)
    cfg = {
        "target": args.target,
        "base_motif": os.path.abspath(args.base_motif),
        "a_cat_json": os.path.abspath(args.a_cat),
        "k": args.k, "n_designs_per_particle": args.n, "n_waves": args.waves,
        "jitter_sigma_A_initial": args.sigma_init,
        "jitter_sigma_A_diversification": args.sigma_div,
        "lambda_smc": args.lam,
        "rfd2_command_template": args.rfd2_template,
        "dry_run": args.dry_run,
        "current_wave": 0,
    }
    json.dump(cfg, open(os.path.join(wd, "config.json"), "w"), indent=2)

    # wave 0: K particles jittered from base motif
    for k in range(args.k):
        pdir = particle_dir(wd, 0, k)
        out_motif = os.path.join(pdir, "motif.pdb")
        translations = jitter_pdb(cfg["base_motif"], out_motif,
                                  sigma_A=args.sigma_init, seed=1000 + k)
        json.dump({"wave": 0, "particle": k, "seed": 1000 + k,
                   "sigma_A": args.sigma_init,
                   "translations": {f"{c}{r}": list(t) for (c, r), t in translations.items()}},
                  open(os.path.join(pdir, "particle.json"), "w"), indent=2)
        # write a single sbatch per particle (RFD2 array job)
        sbatch_path = os.path.join(pdir, "submit_rfd2.sbatch")
        with open(sbatch_path, "w") as f:
            f.write(_render_sbatch(cfg, k, 0, pdir))

    print(f"# init: workdir={wd}  K={args.k}  N={args.n}  waves={args.waves}")
    print(f"# wave 0: {args.k} particles ready. To launch:")
    for k in range(args.k):
        print(f"   sbatch {particle_dir(wd, 0, k)}/submit_rfd2.sbatch")
    print(f"# After all particles finish, run:")
    print(f"   python stage1_outer_loop_smc.py score-wave --workdir {wd} --wave 0")
    print(f"   python stage1_outer_loop_smc.py resample   --workdir {wd} --wave 0")


def score_wave(args):
    wd = args.workdir
    cfg = json.load(open(os.path.join(wd, "config.json")))
    wave = args.wave
    fields = ACatFields(json.load(open(cfg["a_cat_json"])))
    particle_summaries = []
    for k in range(cfg["k"]):
        pdir = particle_dir(wd, wave, k)
        # find design PDBs (RFD2 outputs); fall back to motif if no designs
        designs = sorted(p for p in _list_design_pdbs(pdir))
        scores = []
        for p in designs:
            atoms = _parse_pdb(p)
            E, br = e_cat(atoms, fields, return_breakdown=True)
            scores.append({"file": os.path.basename(p), "E_cat": round(E, 4),
                           "terms": br["terms"]})
        if not scores:
            print(f"   particle {k}: NO designs found in {pdir}")
            particle_summaries.append({"particle": k, "n_designs": 0,
                                       "E_cat_mean": None, "E_cat_min": None})
            continue
        E_vals = [s["E_cat"] for s in scores]
        summary = {
            "particle": k, "n_designs": len(scores),
            "E_cat_mean": round(sum(E_vals)/len(E_vals), 4),
            "E_cat_min":  round(min(E_vals), 4),
            "E_cat_median": round(sorted(E_vals)[len(E_vals)//2], 4),
        }
        json.dump({"summary": summary, "designs": scores},
                  open(os.path.join(pdir, "e_cat_scores.json"), "w"), indent=2)
        particle_summaries.append(summary)
    out = os.path.join(wd, f"wave_{wave:03d}", "wave_summary.json")
    json.dump(particle_summaries, open(out, "w"), indent=2)
    print(f"# scored wave {wave}: {len(particle_summaries)} particles")
    for s in particle_summaries:
        print(f"   particle {s['particle']:3d}: n={s['n_designs']:3d}  "
              f"E_min={s['E_cat_min']}  E_mean={s['E_cat_mean']}")


def resample(args):
    wd = args.workdir
    cfg = json.load(open(os.path.join(wd, "config.json")))
    wave = args.wave
    summaries = json.load(open(os.path.join(wd, f"wave_{wave:03d}", "wave_summary.json")))
    valid = [s for s in summaries if s["E_cat_min"] is not None]
    if not valid:
        print(f"# wave {wave}: no valid particles to resample from")
        return
    # weights = exp(-lambda * (E_cat_min - min over particles))
    # use min as the summary statistic (the best design from each particle)
    E_min = [s["E_cat_min"] for s in valid]
    E_offset = min(E_min)
    lam = cfg["lambda_smc"]
    weights = [math.exp(-lam * (e - E_offset)) for e in E_min]
    norm = sum(weights)
    weights_norm = [w / norm for w in weights]
    effective_size = 1.0 / sum(w*w for w in weights_norm)
    print(f"# wave {wave} weights (lambda={lam}):")
    for s, w in zip(valid, weights_norm):
        print(f"   particle {s['particle']:3d}: E_min={s['E_cat_min']}  w={w:.4f}")
    print(f"# effective sample size ESS = {effective_size:.2f} / {len(valid)}")
    if effective_size < 0.3 * len(valid):
        print(f"# WARN: ESS too low ({effective_size:.2f}) - particles collapsing; "
              f"consider lower lambda or more diversification")

    # resample for next wave
    next_wave = wave + 1
    if next_wave >= cfg["n_waves"]:
        print(f"# reached final wave; not resampling.")
        return
    rng = random.Random(cfg.get("resample_seed", 12345) + wave)
    chosen = rng.choices(valid, weights=weights_norm, k=cfg["k"])
    div = cfg["jitter_sigma_A_diversification"]
    for k_next, src in enumerate(chosen):
        # source particle's motif (after its wave-(wave) jitter)
        src_motif = os.path.join(particle_dir(wd, wave, src["particle"]), "motif.pdb")
        pdir = particle_dir(wd, next_wave, k_next)
        out_motif = os.path.join(pdir, "motif.pdb")
        translations = jitter_pdb(src_motif, out_motif, sigma_A=div,
                                  seed=2000 + next_wave * 1000 + k_next)
        json.dump({"wave": next_wave, "particle": k_next,
                   "resampled_from_wave": wave, "resampled_from_particle": src["particle"],
                   "src_E_cat_min": src["E_cat_min"],
                   "diversification_sigma_A": div,
                   "translations": {f"{c}{r}": list(t) for (c, r), t in translations.items()}},
                  open(os.path.join(pdir, "particle.json"), "w"), indent=2)
        sbatch_path = os.path.join(pdir, "submit_rfd2.sbatch")
        with open(sbatch_path, "w") as f:
            f.write(_render_sbatch(cfg, k_next, next_wave, pdir))
    print(f"# wrote wave {next_wave}: {cfg['k']} particles ready")
    print(f"# To launch wave {next_wave}:")
    for k in range(cfg["k"]):
        print(f"   sbatch {particle_dir(wd, next_wave, k)}/submit_rfd2.sbatch")


def harvest(args):
    wd = args.workdir
    cfg = json.load(open(os.path.join(wd, "config.json")))
    rows = []
    for wave in range(cfg["n_waves"]):
        for k in range(cfg["k"]):
            scores_path = os.path.join(particle_dir(wd, wave, k), "e_cat_scores.json")
            if not os.path.isfile(scores_path):
                continue
            data = json.load(open(scores_path))
            for d in data.get("designs", []):
                rows.append({"wave": wave, "particle": k,
                             "file": d["file"], "E_cat": d["E_cat"],
                             "terms": d.get("terms", {}),
                             "dir": particle_dir(wd, wave, k)})
    if not rows:
        print(f"# no scored designs found in {wd}")
        return
    rows.sort(key=lambda r: r["E_cat"])
    top = rows[:args.top]
    out = args.out or os.path.join(wd, "harvest.json")
    json.dump({"target": cfg["target"], "n_total": len(rows),
               "n_top": len(top), "top": top}, open(out, "w"), indent=2)
    print(f"# harvested {len(rows)} scored designs from {cfg['n_waves']} waves, k={cfg['k']}")
    print(f"# top {len(top)} by E_cat:")
    for r in top[:min(len(top), 15)]:
        print(f"   wave {r['wave']} particle {r['particle']:3d}: E_cat={r['E_cat']:.4f}  "
              f"path={r['terms'].get('path',0):.3f}  contact={r['terms'].get('contact',0):.3f}  "
              f"avoid={r['terms'].get('avoid',0):.3f}  site={r['terms'].get('site',0):.3f}")
    print(f"# wrote {out}")


def _list_design_pdbs(d):
    import glob
    pdbs = sorted(p for p in glob.glob(os.path.join(d, "designs", "*-atomized-bb-False.pdb")))
    if not pdbs:
        pdbs = sorted(p for p in glob.glob(os.path.join(d, "*-atomized-bb-False.pdb")))
    return pdbs


def _parse_pdb(path):
    atoms = []
    for line in open(path):
        if line[:6].strip() not in ("ATOM", "HETATM"): continue
        try:
            name = line[12:16].strip()
            el = line[76:78].strip().upper() or "".join(c for c in name if c.isalpha())[:2].upper()
            atoms.append({
                "record": line[:6].strip(), "name": name, "element": el,
                "resname": line[17:20].strip(), "chain": line[21].strip(),
                "resseq": int(line[22:26]),
                "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
            })
        except ValueError:
            continue
    return atoms


def _render_sbatch(cfg, particle, wave, pdir):
    template = cfg.get("rfd2_command_template") or _DEFAULT_RFD2_TEMPLATE
    pdir_posix = pdir.replace("\\", "/")
    return template.format(
        wave=wave, particle=particle,
        n=cfg["n_designs_per_particle"],
        motif=f"{pdir_posix}/motif.pdb",
        output_dir=f"{pdir_posix}/designs",
        target=cfg["target"],
    )


_DEFAULT_RFD2_TEMPLATE = """#!/usr/bin/env bash
# Stage-1 SMC particle: wave {wave}, particle {particle}, target {target}.
# Mirrors cluster/arm_generate_array.sbatch invocation pattern (cd $RFD2, --home
# $RFD2HOME, MKL/DGL env vars, sandbox at rf_diffusion/exec/...sandbox relative
# to $RFD2).
#SBATCH --job-name=smc_w{wave}_p{particle}
#SBATCH --partition=gpu,beta,dgxlo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output={output_dir}/slurm-%j.out
set -uo pipefail

RFD2="${{RFD2_DIR:-/resnick/scratch/atiwari2/RFdiffusion2}}"
REPO="${{REPO_DIR:-/resnick/scratch/atiwari2/rfdiffusion}}"
SCRATCH="${{SCRATCH_DIR:-/resnick/scratch/atiwari2}}"
APPT_MODULE="${{APPT_MODULE:-apptainer/1.3.3-gcc-13.2.0-i5n6b74}}"
export APPTAINER_CACHEDIR="${{APPTAINER_CACHEDIR:-$SCRATCH/.apptainer_cache}}"
export APPTAINER_TMPDIR="$SCRATCH/apptmp/smc_${{SLURM_JOB_ID:-j}}_w{wave}_p{particle}"
RFD2HOME="$SCRATCH/rfd2home"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$RFD2HOME" {output_dir}
trap 'rm -rf "$APPTAINER_TMPDIR"' EXIT
module load "$APPT_MODULE" 2>/dev/null || module load apptainer 2>/dev/null || true

MOTIF="{motif}"
MANIFEST="$REPO/pipeline/compiled/{target}/manifest.json"
[ -f "$MOTIF" ]    || {{ echo "ERROR: motif $MOTIF missing"; exit 1; }}
[ -f "$MANIFEST" ] || {{ echo "ERROR: manifest $MANIFEST missing"; exit 1; }}

CONTIG="$(python3 -c "import json;print(json.load(open('$MANIFEST'))['contig'])")"
CONTIG_ATOMS="$(python3 -c "import json;print(json.load(open('$MANIFEST'))['contig_atoms'])")"

cd "$RFD2" || exit 1
export PYTHONPATH="$PWD"
IMG="rf_diffusion/exec/bakerlab_rf_diffusion_aa_sandbox"
[ -e "$IMG/bin/sh" ] || {{ echo "ERROR: sandbox $RFD2/$IMG missing/corrupt"; exit 1; }}

echo "=== SMC particle wave={wave} particle={particle} target={target} N={n} ==="
echo "    motif       : $MOTIF (jittered from base)"
echo "    output      : {output_dir}"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

apptainer exec --nv -B /resnick --home "$RFD2HOME" \\
  --env DGLBACKEND=pytorch --env MKL_THREADING_LAYER=GNU --env MKL_SERVICE_FORCE_INTEL=1 \\
  "$IMG" \\
  rf_diffusion/run_inference.py --config-name=aa \\
    inference.deterministic=False \\
    inference.ckpt_path=REPO_ROOT/rf_diffusion/model_weights/RFD_173.pt \\
    inference.input_pdb="$MOTIF" \\
    inference.ligand=LIG \\
    contigmap.contigs="$CONTIG" inference.contig_as_guidepost=True \\
    contigmap.contig_atoms="$CONTIG_ATOMS" \\
    inference.num_designs={n} inference.design_startnum=0 \\
    inference.output_prefix={output_dir}/{target}_smc_w{wave}_p{particle} \\
    hydra.job_logging.root.level=WARN
rc=$?
echo "=== particle exit $rc ==="
echo "designs in {output_dir}: $(find {output_dir} -maxdepth 1 -name '*-atomized-bb-False.pdb' | wc -l)"
exit $rc
"""


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    p_init = sp.add_parser("init", help="set up SMC run with wave-0 particles")
    p_init.add_argument("--target", required=True)
    p_init.add_argument("--base-motif", required=True)
    p_init.add_argument("--a-cat", required=True, help="A_cat JSON for E_cat scoring")
    p_init.add_argument("--workdir", required=True)
    p_init.add_argument("--k", type=int, default=8, help="number of particles per wave")
    p_init.add_argument("--n", type=int, default=50, help="designs per particle per wave")
    p_init.add_argument("--waves", type=int, default=3)
    p_init.add_argument("--sigma-init", type=float, default=0.5,
                        help="initial jitter sigma (A) for wave-0 diversification")
    p_init.add_argument("--sigma-div", type=float, default=0.3,
                        help="per-wave diversification sigma (A)")
    p_init.add_argument("--lam", type=float, default=1.0,
                        help="lambda in weight = exp(-lam * (E_cat_min - min))")
    p_init.add_argument("--rfd2-template", default=None,
                        help="path to a custom RFD2 sbatch template")
    p_init.add_argument("--dry-run", action="store_true",
                        help="just set up files; do not submit (always-on for local builds)")
    p_init.set_defaults(func=init_run)

    p_score = sp.add_parser("score-wave", help="score all particles in a wave with E_cat")
    p_score.add_argument("--workdir", required=True)
    p_score.add_argument("--wave", type=int, required=True)
    p_score.set_defaults(func=score_wave)

    p_har = sp.add_parser("harvest", help="collect best designs across all waves by E_cat")
    p_har.add_argument("--workdir", required=True)
    p_har.add_argument("--top", type=int, default=20, help="top-K designs to surface")
    p_har.add_argument("--out", default=None, help="write JSON manifest (default workdir/harvest.json)")
    p_har.set_defaults(func=harvest)

    p_res = sp.add_parser("resample", help="weighted resample to next wave")
    p_res.add_argument("--workdir", required=True)
    p_res.add_argument("--wave", type=int, required=True)
    p_res.set_defaults(func=resample)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
