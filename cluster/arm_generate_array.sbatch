#!/usr/bin/env bash
# arm_generate_array.sbatch — scaled RFD2 generation on a compiled ArM motif, parallel.
# A SLURM array splits the designs across dgxlo V100s (design_startnum offset per task),
# so 100 designs finish in ~wall-clock minutes instead of ~22h serial. Same script runs
# the real target or a Rev3 control (pass the tagged dir, e.g. 3ZP9__wrong_metal).
#
#   sbatch cluster/arm_generate_array.sbatch 3ZP9                      # 20 tasks x 5 = 100
#   sbatch cluster/arm_generate_array.sbatch 3ZP9__wrong_metal
#   DESIGNS_PER_TASK=5 sbatch --array=0-19 cluster/arm_generate_array.sbatch 3ZP9
#SBATCH --job-name=arm_arr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-19%20
#SBATCH --output=arm_arr_%A_%a.out
# H200 on the 'gpu' partition (~80 available) -> all 20 tasks run at once.
# If submission is rejected for the account, add: sbatch --account=<acct> ...
# To fall back to the instant V100s: sbatch --partition=dgxlo --gres=gpu:1 ...
set -uo pipefail

TARGET="${1:-3ZP9}"                       # tagged dir under pipeline/compiled/
PER="${DESIGNS_PER_TASK:-5}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
START=$(( TASK * PER ))

RFD2="${RFD2_DIR:-/resnick/scratch/atiwari2/RFdiffusion2}"
REPO="${REPO_DIR:-/resnick/scratch/atiwari2/rfdiffusion}"
SCRATCH="${SCRATCH_DIR:-/resnick/scratch/atiwari2}"
APPT_MODULE="${APPT_MODULE:-apptainer/1.3.3-gcc-13.2.0-i5n6b74}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH/.apptainer_cache}"
# Per-task TMPDIR: concurrent apptainer execs of the sandbox collide on a shared session
# dir (-> "/bin/sh not found / container empty"). Isolate each array task.
export APPTAINER_TMPDIR="$SCRATCH/apptmp/${SLURM_JOB_ID:-j}_${SLURM_ARRAY_TASK_ID:-0}"
RFD2HOME="$SCRATCH/rfd2home"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$RFD2HOME"
trap 'rm -rf "$APPTAINER_TMPDIR"' EXIT
module load "$APPT_MODULE" 2>/dev/null || module load apptainer 2>/dev/null || true

ARM="$REPO/pipeline/compiled"
MOTIF="$ARM/$TARGET/motif.pdb"
MANIFEST="$ARM/$TARGET/manifest.json"
[ -f "$MOTIF" ] && [ -f "$MANIFEST" ] || { echo "ERROR: compiled/$TARGET missing — git pull"; exit 1; }

CONTIG="$(python3 -c "import json;print(json.load(open('$MANIFEST'))['contig'])")"
CONTIG_ATOMS="$(python3 -c "import json;print(json.load(open('$MANIFEST'))['contig_atoms'])")"
OUTDIR="$SCRATCH/RFdiffusion2/arm_designs/$TARGET"
OUTPREF="$OUTDIR/${TARGET}_cond0"
mkdir -p "$OUTDIR"

cd "$RFD2" || exit 1
export PYTHONPATH="$PWD"
SANDBOX="rf_diffusion/exec/bakerlab_rf_diffusion_aa_sandbox"

echo "=== $(hostname) | array task $TASK | designs $START..$((START+PER-1)) | TARGET=$TARGET ==="
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

apptainer exec --nv -B /resnick --home "$RFD2HOME" \
  --env DGLBACKEND=pytorch --env MKL_THREADING_LAYER=GNU --env MKL_SERVICE_FORCE_INTEL=1 \
  "$SANDBOX" \
  rf_diffusion/run_inference.py --config-name=aa \
    inference.deterministic=True \
    inference.ckpt_path=REPO_ROOT/rf_diffusion/model_weights/RFD_173.pt \
    inference.input_pdb="$MOTIF" inference.ligand=LIG \
    contigmap.contigs="$CONTIG" inference.contig_as_guidepost=True \
    contigmap.contig_atoms="$CONTIG_ATOMS" \
    inference.num_designs="$PER" inference.design_startnum="$START" \
    inference.output_prefix="$OUTPREF" hydra.job_logging.root.level=WARN
echo "=== task $TASK exit $? ==="
echo "designs so far in $OUTDIR: $(find "$OUTDIR" -maxdepth 1 -name '*-atomized-bb-False.pdb' | wc -l)"
