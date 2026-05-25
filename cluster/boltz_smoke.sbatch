#!/usr/bin/env bash
# boltz_smoke.sbatch — confirm Boltz-2 (metal-aware) runs on a Caltech V100.
# This single test decides whether the whole pipeline stays on free Caltech GPUs
# (success) or we need a Caltech H100 / RunPod H200 for folding (failure).
# Submit AFTER setup_boltz.sh:  sbatch boltz_smoke.sbatch
#SBATCH --job-name=boltz_smoke
#SBATCH --partition=dgxlo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=boltz_smoke_%j.out
set -uo pipefail

SCRATCH="${SCRATCH_DIR:-/resnick/scratch/atiwari2}"
VENV="${BOLTZ_VENV:-$SCRATCH/boltz-venv}"
CLUSTER_DIR="${CLUSTER_DIR:-/resnick/scratch/atiwari2/rfdiffusion/cluster}"
export BOLTZ_CACHE="${BOLTZ_CACHE:-$SCRATCH/.boltz}"

module load python/3.11.6 2>/dev/null || module load python/3.11 2>/dev/null || module load python3 2>/dev/null || true
# shellcheck disable=SC1091
source "$VENV/bin/activate" || { echo "ERROR: venv $VENV not found — run setup_boltz.sh first"; exit 1; }

echo "=== host $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

echo
echo "=== Boltz-2 V100 smoke: ubiquitin + Zn, single-sequence (msa: empty), --no_kernels ==="
echo "    (--no_kernels disables cuequivariance, required on compute-7.0 V100)"
time boltz predict "$CLUSTER_DIR/boltz_smoke.yaml" \
  --cache "$BOLTZ_CACHE" --out_dir "$SCRATCH/boltz_smoke_out" \
  --accelerator gpu --devices 1 --no_kernels --override
echo "=== boltz exit code: $? ==="

echo
echo "=== outputs ==="
ls -ltR "$SCRATCH/boltz_smoke_out" 2>/dev/null | head -40
echo
echo "SUCCESS = a predicted .cif/.pdb structure under boltz_smoke_out/ (a confidence .json too)."
echo "That proves metal-aware Boltz-2 runs on the V100 -> the entire pipeline stays on free"
echo "Caltech GPUs; no RunPod, no AF3 weights gate, no H100 queue needed."
echo "If it OOMs or errors on the kernels, tell me — first fallback is a Caltech H100 (beta/gpu)."
