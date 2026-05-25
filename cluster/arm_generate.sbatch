#!/usr/bin/env bash
# arm_generate.sbatch — run RFD2 on a COMPILED ArM motif (Sub-question B generation).
# Uses run_inference.py directly (not the benchmark pipeline) on our cofactor reactive core.
# First run with NUM_DESIGNS=1 to confirm RFD2 INGESTS the organometallic core (Ir + donors);
# if it reaches "Flow-matching step: 1/100", ingestion worked -> scale to 100 + controls.
#
#   sbatch cluster/arm_generate.sbatch 3ZP9            # 1-design ingestion test (default)
#   NUM_DESIGNS=100 sbatch cluster/arm_generate.sbatch 3ZP9   # full run
#SBATCH --job-name=arm_gen
#SBATCH --partition=dgxlo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=arm_gen_%j.out
set -uo pipefail

TARGET="${1:-3ZP9}"
NUM="${NUM_DESIGNS:-1}"
SCAFFOLD="${SCAFFOLD_LENGTH:-160}"
RFD2="${RFD2_DIR:-/resnick/scratch/atiwari2/RFdiffusion2}"
REPO="${REPO_DIR:-/resnick/scratch/atiwari2/rfdiffusion}"
SCRATCH="${SCRATCH_DIR:-/resnick/scratch/atiwari2}"
APPT_MODULE="${APPT_MODULE:-apptainer/1.3.3-gcc-13.2.0-i5n6b74}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH/.apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$SCRATCH/apptmp}"
RFD2HOME="$SCRATCH/rfd2home"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$RFD2HOME"
module load "$APPT_MODULE" 2>/dev/null || module load apptainer 2>/dev/null || true

ARM="$REPO/pipeline/compiled"
MOTIF="$ARM/$TARGET/motif.pdb"
MANIFEST="$ARM/$TARGET/manifest.json"
[ -f "$MOTIF" ]    || { echo "ERROR: $MOTIF missing — git pull the repo (pipeline/compiled/$TARGET/)"; exit 1; }
[ -f "$MANIFEST" ] || { echo "ERROR: $MANIFEST missing"; exit 1; }

CORE="$(python3 -c "import json;print(','.join(json.load(open('$MANIFEST'))['core_atoms']))")"
OUTDIR="$SCRATCH/RFdiffusion2/arm_designs/$TARGET"
OUTPREF="$OUTDIR/${TARGET}_cond0"
mkdir -p "$OUTDIR"

cd "$RFD2" || { echo "ERROR: $RFD2 missing — run setup_rfd2.sh"; exit 1; }
export PYTHONPATH="$PWD"
SANDBOX="rf_diffusion/exec/bakerlab_rf_diffusion_aa_sandbox"
[ -d "$SANDBOX" ] || { echo "ERROR: $SANDBOX missing — run extract_sif.sh"; exit 1; }

echo "=== host $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo "=== ArM generation: TARGET=$TARGET  NUM_DESIGNS=$NUM  SCAFFOLD=$SCAFFOLD ==="
echo "    motif: $MOTIF"
echo "    rigid cofactor core (ligand LIG): $CORE"
echo "    MODE: ligand-only buried pocket  contig=['$SCAFFOLD']  rasa=0"
echo "          (mirrors RFD2's small_molecule_binder_rasa_buried / heme entries; NO fixed"
echo "           protein residues; whole cofactor core is the rigid ligand)."
echo "    INGESTION CHECK: reaching 'Flow-matching step: 1/...' proves RFD2 atomized the"
echo "    organometallic core and is building a pocket around it."
echo

apptainer exec --nv -B /resnick --home "$RFD2HOME" \
  --env DGLBACKEND=pytorch \
  --env MKL_THREADING_LAYER=GNU \
  --env MKL_SERVICE_FORCE_INTEL=1 \
  "$SANDBOX" \
  rf_diffusion/run_inference.py --config-name=aa \
    inference.deterministic=True \
    inference.ckpt_path=REPO_ROOT/rf_diffusion/model_weights/RFD_173.pt \
    inference.input_pdb="$MOTIF" \
    inference.ligand=LIG \
    contigmap.contigs="['${SCAFFOLD}']" \
    contigmap.length="${SCAFFOLD}-${SCAFFOLD}" \
    inference.conditions.relative_sasa_v2.active=True \
    inference.conditions.relative_sasa_v2.rasa=0 \
    inference.num_designs="$NUM" inference.design_startnum=0 \
    inference.output_prefix="$OUTPREF" \
    hydra.job_logging.root.level=WARN
echo "=== run_inference exit code: $? ==="

echo
echo "=== outputs ==="
ls -ltR "$OUTDIR" 2>/dev/null | head -40
find "$OUTDIR" -name '*.pdb' 2>/dev/null | head
echo
echo "SUCCESS = a design .pdb under $OUTDIR (a protein backbone built around the cofactor core)."
echo "That validates the motif compiler + reactive-core representation end-to-end on a real ArM."
