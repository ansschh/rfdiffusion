#!/usr/bin/env bash
# stage2_chemistry_run.sh - Stage 2c real-vs-damaged A_cat comparison.
# Run only AFTER stage2_vanilla_preservation_test.sh passes.
#
# Spawns 4 SBATCH jobs (one per variant): real, inverted_face, flipped_face,
# wrong_hapticity. Each runs K=4 particles with lambda guidance. Cross-eval
# at the end scores every variant's outputs under REAL A_cat.

set -uo pipefail

TARGET="${1:?target required, e.g. 3ZP9}"
K="${2:-4}"
LAM="${3:-1.0}"
MODE="${4:-tiered}"           # tiered | linear  (lambda schedule)
ACAT_MODE="${5:-oracle}"      # oracle | chem    (A_cat source)

RFD2="${RFD2_DIR:-/resnick/scratch/atiwari2/RFdiffusion2}"
REPO="${REPO_DIR:-/resnick/scratch/atiwari2/rfdiffusion}"
SCRATCH="${SCRATCH_DIR:-/resnick/scratch/atiwari2}"

COMPILED="$REPO/pipeline/compiled/$TARGET"
if [ "$ACAT_MODE" = "chem" ]; then
    ACAT="$COMPILED/A_cat_chem.json"
    DMG_DIR="$COMPILED/damaged_chem"
    OUT_SUFFIX="_chem"
    [ -f "$ACAT" ] || python3 "$REPO/pipeline/retrieval/instantiate_acat.py" "$COMPILED" --mode chem
else
    ACAT="$COMPILED/A_cat.json"
    DMG_DIR="$COMPILED/damaged"
    OUT_SUFFIX=""
    [ -f "$ACAT" ] || python3 "$REPO/pipeline/retrieval/instantiate_acat.py" "$COMPILED" --mode oracle
fi
echo "# ACAT_MODE=$ACAT_MODE  source=$ACAT  out_suffix='$OUT_SUFFIX'"

# Generate damaged A_cat variants (per-mode dir, doesn't overwrite oracle)
python3 "$REPO/pipeline/guidance/damaged_controls.py" "$ACAT" --out-dir "$DMG_DIR"

CONTIG="$(python3 -c "import json;print(json.load(open('$COMPILED/manifest.json'))['contig'])")"
CONTIG_ATOMS="$(python3 -c "import json;print(json.load(open('$COMPILED/manifest.json'))['contig_atoms'])")"

TIERED_FLAG=""
[ "$MODE" = "tiered" ] && TIERED_FLAG="--tiered-lambdas"

JIDS=()
for VARIANT in real inverted_face flipped_face wrong_hapticity; do
    A="$DMG_DIR/$VARIANT.json"
    [ "$VARIANT" = "real" ] && A="$ACAT"
    OUT="$SCRATCH/RFdiffusion2/smc2/${TARGET}_${VARIANT}_K${K}_L${LAM}_${MODE}${OUT_SUFFIX}"
    mkdir -p "$OUT"

    SBATCH_FILE="$OUT/submit.sbatch"
    cat > "$SBATCH_FILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=smc2_${VARIANT}
#SBATCH --partition=gpu,beta
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=$OUT/slurm-%j.out
set -uo pipefail

export APPTAINER_CACHEDIR="$SCRATCH/.apptainer_cache"
export APPTAINER_TMPDIR="$SCRATCH/apptmp/smc2_${VARIANT}_\${SLURM_JOB_ID}"
mkdir -p "\$APPTAINER_TMPDIR" "$SCRATCH/rfd2home"
trap 'rm -rf "\$APPTAINER_TMPDIR"' EXIT
module load apptainer/1.3.3-gcc-13.2.0-i5n6b74 2>/dev/null || module load apptainer 2>/dev/null || true

cd "$RFD2"
export PYTHONPATH="\$PWD:$REPO/pipeline"
IMG="rf_diffusion/exec/bakerlab_rf_diffusion_aa_sandbox"

apptainer exec --nv -B /resnick --home "$SCRATCH/rfd2home" \\
  --env DGLBACKEND=pytorch --env MKL_THREADING_LAYER=GNU --env MKL_SERVICE_FORCE_INTEL=1 \\
  --env REPO_DIR="$REPO" --env RFD2_DIR="$RFD2" \\
  "\$IMG" \\
  python "$REPO/cluster/stage2_rfd2_hook_v173.py" \\
    --motif "$COMPILED/motif.pdb" \\
    --a-cat "$A" \\
    --ligand LIG \\
    --contigs "$CONTIG" \\
    --contig-atoms "$CONTIG_ATOMS" \\
    --ckpt REPO_ROOT/rf_diffusion/model_weights/RFD_173.pt \\
    --out-dir "$OUT" \\
    -K "$K" --checkpoint-every 10 --lambda-max "$LAM" \\
    $TIERED_FLAG --seed 42
EOF
    JID=$(sbatch --parsable "$SBATCH_FILE")
    echo "variant=$VARIANT  job=$JID  out=$OUT"
    JIDS+=("$JID")
done

# Wait for all variants to finish
JOB_LIST=$(IFS=,; echo "${JIDS[*]}")
echo ""
echo "# waiting on jobs: $JOB_LIST"
while squeue -j "$JOB_LIST" -h -o "%j" 2>/dev/null | grep -q .; do
    sleep 120
done
echo "# all variants complete"

# Cross-evaluate: score each variant's outputs against REAL A_cat
echo ""
echo "=== CROSS-EVALUATION: all variants scored under REAL A_cat ==="
for VARIANT in real inverted_face flipped_face wrong_hapticity; do
    OUT="$SCRATCH/RFdiffusion2/smc2/${TARGET}_${VARIANT}_K${K}_L${LAM}_${MODE}${OUT_SUFFIX}"
    echo ""
    echo "--- variant: $VARIANT ---"
    python3 "$REPO/pipeline/v_chem.py" "$OUT" 2>&1 | tail -8
    python3 "$REPO/pipeline/v_rxn.py" "$OUT" 2>&1 | tail -8
done
echo ""
echo "# Stage 2c verdict: if 'real' variant has measurably higher V_chem/V_rxn"
echo "# pass rates than damaged variants, guided generation is chemistry-faithful."
