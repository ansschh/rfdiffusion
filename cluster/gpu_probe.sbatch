#!/usr/bin/env bash
# gpu_probe.sbatch — tiny real GPU job: confirms you can get a GPU and reports its specs.
# The partition is injected on the command line by run_probe_and_time.sh (overrides any here).
# Standalone use:  sbatch --partition=<gpu_part> gpu_probe.sbatch
#SBATCH --job-name=gpuprobe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=gpuprobe_%j.out
set -uo pipefail
echo "=== HOST: $(hostname)   START: $(date) ==="
echo
echo "=== SLURM job environment ==="
env | grep -E '^SLURM_(JOB_ID|JOB_PARTITION|JOB_NODELIST|JOB_QOS|JOB_ACCOUNT|GPUS|GPUS_ON_NODE|JOB_GPUS|GRES|CPUS_ON_NODE|MEM_PER_NODE)' | sort
echo
echo "=== nvidia-smi (the GPU you were allocated) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  echo
  echo "=== GPU summary (model / VRAM / driver / compute capability) ==="
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
else
  echo "nvidia-smi not found on this compute node (unexpected for a GPU job)."
fi
echo
echo "=== CUDA toolkit in PATH? ==="
nvcc --version 2>/dev/null || echo "nvcc not in PATH — RFdiffusion runs usually need:  module load cuda  (check 'module avail cuda')"
echo
echo "=== visible devices ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo
echo "=== END: $(date) ==="
