#!/usr/bin/env bash
# run_probe_and_time.sh — submit gpu_probe.sbatch and MEASURE the real submit->start wait.
# This is the ground-truth answer to "how long to actually get a GPU right now."
#
# Usage:
#   bash run_probe_and_time.sh                  # auto-pick a GPU partition
#   PROBE_PARTITION=gpu bash run_probe_and_time.sh
#   PROBE_GPUS=1 PROBE_TIME=00:05:00 bash run_probe_and_time.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PART="${PROBE_PARTITION:-}"
GPUS="${PROBE_GPUS:-1}"

# Auto-pick: GPU partition with the most free GPUs right now, else the first GPU partition.
if [[ -z "$PART" ]]; then
  PART=$(sinfo -h -o "%P %G %t %D" \
        | awk 'tolower($2) ~ /gpu/ && $3 ~ /idle|mix/ {gsub(/\*/,"",$1); print $4, $1}' \
        | sort -rn | awk 'NR==1{print $2}')
fi
if [[ -z "$PART" ]]; then
  PART=$(sinfo -h -o "%P %G" | awk 'tolower($2) ~ /gpu/ {gsub(/\*/,"",$1); print $1; exit}')
fi
if [[ -z "$PART" ]]; then
  echo "No GPU partition found automatically. Set PROBE_PARTITION=<name> and retry."
  echo "Candidates:"; sinfo -o "%P %G" | grep -i gpu
  exit 1
fi

echo "Probe target partition: $PART  (requesting --gres=gpu:${GPUS})"
SUBMIT_EPOCH=$(date +%s)
JID=$(sbatch --parsable --partition="$PART" --gres="gpu:${GPUS}" "$HERE/gpu_probe.sbatch") \
  || { echo "sbatch submission failed (access/QOS/gres mismatch?). See cluster_recon Section 1/6."; exit 1; }
JID="${JID%%;*}"
echo "Submitted job $JID at $(date '+%H:%M:%S')."
echo "Polling every 5s until it leaves PENDING (Ctrl-C stops the wait, job keeps running)..."

START_EPOCH=""
while true; do
  ST=$(squeue -h -j "$JID" -o "%T" 2>/dev/null)
  if [[ -z "$ST" ]]; then
    echo; echo "Job $JID no longer in queue (started+finished fast, or failed). Check gpuprobe_${JID}.out"
    break
  fi
  if [[ "$ST" != "PENDING" && "$ST" != "CONFIGURING" ]]; then
    START_EPOCH=$(date +%s)
    echo; echo "Job $JID reached state '$ST'."
    break
  fi
  REASON=$(squeue -h -j "$JID" -o "%r" 2>/dev/null)
  printf '\r  pending %ss   (reason: %s)            ' "$(( $(date +%s) - SUBMIT_EPOCH ))" "${REASON:-?}"
  sleep 5
done

if [[ -n "$START_EPOCH" ]]; then
  echo ">>> MEASURED submit->start wait: $(( START_EPOCH - SUBMIT_EPOCH )) seconds  (partition=$PART, ${GPUS} GPU)"
fi
echo "GPU details will be in:  gpuprobe_${JID}.out   (cat it once the job finishes)"
echo "To cancel:  scancel $JID"
