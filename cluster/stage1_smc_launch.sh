#!/bin/bash
# stage1_smc_launch.sh - launch Stage 1 outer-loop SMC for a target on Caltech HPC.
#
# Workflow:
#   1. Verify sandbox integrity (extract if missing).
#   2. Instantiate A_cat for the target (oracle by default, --chem switch available).
#   3. Initialize SMC workdir with K particles, wave 0 motifs jittered.
#   4. Submit wave 0 RFD2 array; wait for completion.
#   5. Score with E_cat; resample to wave 1.
#   6. Repeat for waves 1, 2.
#   7. Harvest top designs by E_cat.
#
# Requires the env vars: $REPO_DIR, $SCRATCH, $SANDBOX (sandbox dir, not .sif).
# Run from $REPO_DIR after `git pull`.
#
# Usage:
#   bash cluster/stage1_smc_launch.sh 3ZP9 [K=8] [N=50] [waves=3] [mode=oracle]
#
# Example:
#   bash cluster/stage1_smc_launch.sh 3ZP9 8 50 3 oracle

set -uo pipefail

TARGET="${1:?target required, e.g. 3ZP9}"
K="${2:-8}"
N="${3:-50}"
WAVES="${4:-3}"
MODE="${5:-oracle}"

# Path defaults from the working arm_generate_array.sbatch
RFD2="${RFD2_DIR:-/resnick/scratch/atiwari2/RFdiffusion2}"
REPO_DIR="${REPO_DIR:-/resnick/scratch/atiwari2/rfdiffusion}"
SCRATCH="${SCRATCH_DIR:-/resnick/scratch/atiwari2}"
SANDBOX="${SANDBOX:-$RFD2/rf_diffusion/exec/bakerlab_rf_diffusion_aa_sandbox}"
export RFD2_DIR="$RFD2"
export REPO_DIR
export SCRATCH_DIR="$SCRATCH"

WORKDIR="$SCRATCH/RFdiffusion2/smc/${TARGET}_${MODE}_K${K}_N${N}_W${WAVES}"
COMPILED="$REPO_DIR/pipeline/compiled/$TARGET"

# 1. sandbox integrity
if [ ! -e "$SANDBOX/bin/sh" ]; then
    echo "FATAL: SANDBOX broken at $SANDBOX. Re-extract with cluster/extract_sif.sh first."
    exit 1
fi
echo "# sandbox OK at $SANDBOX"

# 2. A_cat
ACAT_FILE="$COMPILED/A_cat.json"
[ "$MODE" = "chem" ] && ACAT_FILE="$COMPILED/A_cat_chem.json"
if [ ! -f "$ACAT_FILE" ]; then
    echo "# instantiating A_cat ($MODE) for $TARGET"
    python "$REPO_DIR/pipeline/retrieval/instantiate_acat.py" "$COMPILED" --mode "$MODE"
fi
echo "# A_cat ready at $ACAT_FILE"

# 3. init SMC workdir
mkdir -p "$WORKDIR"
echo "# workdir: $WORKDIR"
python "$REPO_DIR/pipeline/guidance/stage1_outer_loop_smc.py" init \
    --target "$TARGET" \
    --base-motif "$COMPILED/motif.pdb" \
    --a-cat "$ACAT_FILE" \
    --workdir "$WORKDIR" \
    --k "$K" --n "$N" --waves "$WAVES" \
    --sigma-init 0.5 --sigma-div 0.3 --lam 1.0

# 4-6. wave-by-wave loop
for WAVE in $(seq 0 $((WAVES-1))); do
    echo ""
    echo "=== WAVE $WAVE: submitting $K particles ==="
    WAVE_DIR="$WORKDIR/wave_$(printf '%03d' $WAVE)"
    JIDS=()
    for k in $(seq 0 $((K-1))); do
        P_DIR="$WAVE_DIR/particle_$(printf '%03d' $k)"
        JID=$(sbatch --parsable "$P_DIR/submit_rfd2.sbatch")
        echo "   particle $k: jobid $JID"
        JIDS+=("$JID")
    done

    # wait for all particle jobs to finish
    JOB_LIST=$(IFS=,; echo "${JIDS[*]}")
    echo "# waiting on jobs: $JOB_LIST"
    while squeue -j "$JOB_LIST" -h -o "%j" 2>/dev/null | grep -q .; do
        sleep 60
    done
    echo "# wave $WAVE jobs complete"

    # score the wave
    python "$REPO_DIR/pipeline/guidance/stage1_outer_loop_smc.py" score-wave \
        --workdir "$WORKDIR" --wave $WAVE

    # resample to next wave (unless this was the last)
    if [ "$WAVE" -lt $((WAVES-1)) ]; then
        python "$REPO_DIR/pipeline/guidance/stage1_outer_loop_smc.py" resample \
            --workdir "$WORKDIR" --wave $WAVE
    fi
done

# 7. harvest
echo ""
echo "=== HARVEST ==="
python "$REPO_DIR/pipeline/guidance/stage1_outer_loop_smc.py" harvest \
    --workdir "$WORKDIR" --top 30

echo ""
echo "# Stage 1 SMC complete for $TARGET. Workdir: $WORKDIR"
echo "# Inspect: $WORKDIR/harvest.json"
