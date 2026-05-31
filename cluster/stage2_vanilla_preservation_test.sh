#!/usr/bin/env bash
# stage2_vanilla_preservation_test.sh - GATE check for Stage 2.
#
# Runs the RFD2 hook with --K 1 --lambda-max 0 --deterministic on 3ZP9.
# Output should be identical (or near-identical) to a baseline RFD2 run at
# the same seed. If V_chem/V_rxn pass rates match baseline within tolerance,
# the hook is safe; proceed to chemistry runs.
#
# Mirrors arm_generate_array.sbatch invocation pattern (cd $RFD2, --home,
# MKL/DGL env vars, sandbox at rf_diffusion/exec/...sandbox).

#SBATCH --job-name=smc2_vanilla
#SBATCH --partition=gpu,beta,dgxlo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=smc2_vanilla_%j.out
set -uo pipefail

TARGET="${1:-3ZP9}"
K="${2:-1}"
RFD2="${RFD2_DIR:-/resnick/scratch/atiwari2/RFdiffusion2}"
REPO="${REPO_DIR:-/resnick/scratch/atiwari2/rfdiffusion}"
SCRATCH="${SCRATCH_DIR:-/resnick/scratch/atiwari2}"
APPT_MODULE="${APPT_MODULE:-apptainer/1.3.3-gcc-13.2.0-i5n6b74}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH/.apptainer_cache}"
export APPTAINER_TMPDIR="$SCRATCH/apptmp/smc2v_${SLURM_JOB_ID:-j}"
RFD2HOME="$SCRATCH/rfd2home"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$RFD2HOME"
trap 'rm -rf "$APPTAINER_TMPDIR"' EXIT
module load "$APPT_MODULE" 2>/dev/null || module load apptainer 2>/dev/null || true

COMPILED="$REPO/pipeline/compiled/$TARGET"
ACAT="$COMPILED/A_cat.json"
[ -f "$ACAT" ] || python3 "$REPO/pipeline/retrieval/instantiate_acat.py" "$COMPILED" --mode oracle
CONTIG="$(python3 -c "import json;print(json.load(open('$COMPILED/manifest.json'))['contig'])")"
CONTIG_ATOMS="$(python3 -c "import json;print(json.load(open('$COMPILED/manifest.json'))['contig_atoms'])")"

OUT="$SCRATCH/RFdiffusion2/smc2/${TARGET}_vanilla_K${K}"
mkdir -p "$OUT"

cd "$RFD2" || exit 1
export PYTHONPATH="$PWD:$REPO/pipeline"
IMG="rf_diffusion/exec/bakerlab_rf_diffusion_aa_sandbox"
[ -e "$IMG/bin/sh" ] || { echo "ERROR: sandbox $RFD2/$IMG missing/corrupt"; exit 1; }

echo "=== Stage 2 vanilla preservation test: K=$K, lambda_max=0, deterministic ==="
echo "=== target=$TARGET out=$OUT ==="
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

apptainer exec --nv -B /resnick --home "$RFD2HOME" \
  --env DGLBACKEND=pytorch --env MKL_THREADING_LAYER=GNU --env MKL_SERVICE_FORCE_INTEL=1 \
  --env REPO_DIR="$REPO" --env RFD2_DIR="$RFD2" \
  "$IMG" \
  python "$REPO/cluster/stage2_rfd2_hook_v173.py" \
    --motif "$COMPILED/motif.pdb" \
    --a-cat "$ACAT" \
    --ligand LIG \
    --contigs "$CONTIG" \
    --contig-atoms "$CONTIG_ATOMS" \
    --ckpt REPO_ROOT/rf_diffusion/model_weights/RFD_173.pt \
    --out-dir "$OUT" \
    -K "$K" --checkpoint-every 10 --lambda-max 0.0 \
    --deterministic --seed 42
rc=$?
echo "=== smc exit $rc ==="

echo ""
echo "=== Output PDBs ==="
find "$OUT" -name "*.pdb" 2>/dev/null | head -10

echo ""
echo "=== V_chem on vanilla output ==="
python3 "$REPO/pipeline/v_chem.py" "$OUT" 2>&1 | tail -15

echo ""
echo "=== V_rxn on vanilla output ==="
python3 "$REPO/pipeline/v_rxn.py" "$OUT" 2>&1 | tail -15

echo ""
echo "=== Baseline 100-design ensemble for comparison ==="
python3 "$REPO/pipeline/v_chem.py" "$SCRATCH/RFdiffusion2/arm_designs/$TARGET" 2>&1 | tail -10 || true
python3 "$REPO/pipeline/v_rxn.py" "$SCRATCH/RFdiffusion2/arm_designs/$TARGET" 2>&1 | tail -10 || true

echo ""
echo "# GATE: V_chem and V_rxn frac_all_pass / frac_access_pass for vanilla SMC"
echo "# should match baseline within ~0.1. If they do, proceed to chemistry run."
exit $rc
