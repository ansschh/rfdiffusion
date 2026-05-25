#!/usr/bin/env bash
# smoke_rfd2.sbatch — prove the RFD2 toolchain end-to-end on an instant dgxlo V100.
# Runs the bundled open_source_demo (one active-site case). Submit AFTER setup_rfd2.sh.
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
APPT_MODULE="${APPT_MODULE:-apptainer/1.3.3-gcc-13.2.0-i5n6b74}"

module load "$APPT_MODULE" 2>/dev/null || module load apptainer 2>/dev/null || true
cd "$RFD2" || { echo "ERROR: $RFD2 not found — run setup_rfd2.sh first"; exit 1; }
export PYTHONPATH="$PWD"

echo "=== host: $(hostname)   $(date) ==="
echo "=== allocated GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

SIF="rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif"
[ -f "$SIF" ] || { echo "ERROR: container $SIF missing — run setup_rfd2.sh first"; exit 1; }

echo
echo "=== RFD2 GENERATION smoke test (one active-site demo case) ==="
echo "    NOTE: the bundled demo's Chai-1 FOLDING sub-step may error on this V100 (compute 7.0)."
echo "    That is EXPECTED and not a failure of RFD2 — backbone PDBs are produced BEFORE that step,"
echo "    and we are replacing Chai-1 with AF2/AF3 for validation anyway. We only need the generation"
echo "    stage to succeed here."
echo
# -B /resnick binds the scratch tree into the container so it can read the repo + weights.
apptainer exec --nv -B /resnick "$SIF" \
  rf_diffusion/benchmark/pipeline.py --config-name=open_source_demo \
  sweep.benchmarks=active_site_unindexed_atomic_partial_ligand
echo "=== pipeline exit code: $? ==="

echo
echo "=== generated outputs (look for *.pdb backbones — that's the success signal) ==="
ls -ltR pipeline_outputs/ 2>/dev/null | head -50
echo
echo "If you see backbone .pdb files under pipeline_outputs/, RFD2 generation works on the V100."
echo "A Chai-1 error after that is the expected, harmless step we route around."
