#!/usr/bin/env bash
# smoke_rfd2.sbatch — prove RFD2 GENERATION on an instant dgxlo V100.
# Runs from the EXTRACTED sandbox dir (the shipped .sif is lz4 and won't mount here).
# Prereq: bash cluster/extract_sif.sh rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif
#   sbatch smoke_rfd2.sbatch
#SBATCH --job-name=rfd2_smoke
#SBATCH --partition=dgxlo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=rfd2_smoke_%j.out
set -uo pipefail

RFD2="${RFD2_DIR:-/resnick/scratch/atiwari2/RFdiffusion2}"
SCRATCH="${SCRATCH_DIR:-/resnick/scratch/atiwari2}"
APPT_MODULE="${APPT_MODULE:-apptainer/1.3.3-gcc-13.2.0-i5n6b74}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH/.apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$SCRATCH/apptmp}"
RFD2HOME="$SCRATCH/rfd2home"   # writable HOME in scratch — real home is over quota and
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$RFD2HOME"   # DGL/Triton/matplotlib/torch-hub all write to ~
module load "$APPT_MODULE" 2>/dev/null || module load apptainer 2>/dev/null || true
cd "$RFD2" || { echo "ERROR: $RFD2 not found — run setup_rfd2.sh first"; exit 1; }
export PYTHONPATH="$PWD"

echo "=== host: $(hostname)   $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

SANDBOX="rf_diffusion/exec/bakerlab_rf_diffusion_aa_sandbox"
[ -d "$SANDBOX" ] || { echo "ERROR: $SANDBOX missing — run: bash cluster/extract_sif.sh $RFD2/rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif"; exit 1; }

echo
echo "=== RFD2 GENERATION smoke (running from extracted sandbox, not the lz4 .sif) ==="
echo "    NOTE: the demo's downstream LigandMPNN/Chai-1 stages may error — their .sif are also"
echo "    lz4 (not yet extracted) and Chai needs a newer GPU. That's fine: we only need RFD2"
echo "    generation to emit backbone .pdb files. Boltz-2 is our validator."
echo
apptainer exec --nv -B /resnick --home "$RFD2HOME" \
  --env DGLBACKEND=pytorch \
  --env MKL_THREADING_LAYER=GNU \
  --env MKL_SERVICE_FORCE_INTEL=1 \
  "$SANDBOX" \
  rf_diffusion/benchmark/pipeline.py --config-name=open_source_demo \
  sweep.benchmarks=active_site_unindexed_atomic_partial_ligand
echo "=== pipeline exit code: $? ==="

echo
echo "=== generated outputs (SUCCESS = backbone .pdb files here) ==="
ls -ltR pipeline_outputs/ 2>/dev/null | head -50
find pipeline_outputs/ -name '*.pdb' 2>/dev/null | head -10
echo
echo "If backbone .pdb files exist, RFD2 generation works on the V100 — generator proven."
