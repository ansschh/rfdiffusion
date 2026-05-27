#!/usr/bin/env bash
# hpc_status.sh — READ-ONLY survey of OrganoEnzymeGen pipeline state on the Caltech HPC.
# Submits nothing. Run it from a login node (or anywhere) to see "where are we actually":
# what's installed in scratch, container/sandbox integrity, how many designs exist per
# target, which scores are computed, running/recent SLURM jobs, and GPU availability.
#
#   bash cluster/hpc_status.sh                 # surveys the default 4 targets + 2 controls
#   bash cluster/hpc_status.sh 3ZP9 5L8D       # only the named targets
#
# Override paths via env (defaults match the generation sbatch scripts):
#   SCRATCH_DIR=/resnick/scratch/atiwari2  RFD2_DIR=$SCRATCH/RFdiffusion2  REPO_DIR=$SCRATCH/rfdiffusion
set -uo pipefail

SCRATCH="${SCRATCH_DIR:-/resnick/scratch/atiwari2}"
RFD2="${RFD2_DIR:-$SCRATCH/RFdiffusion2}"
REPO="${REPO_DIR:-$SCRATCH/rfdiffusion}"
ARM="$RFD2/arm_designs"
SC="$SCRATCH/selfconsist"
SANDBOX="$RFD2/rf_diffusion/exec/bakerlab_rf_diffusion_aa_sandbox"
SIF="$RFD2/rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif"
WEIGHTS="$RFD2/rf_diffusion/model_weights/RFD_173.pt"

TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(3ZP9 3WJC 5L8D 5OD5 3ZP9__wrong_metal 3ZP9__scramble_guideposts)

hr(){ printf '%s\n' "------------------------------------------------------------"; }
yn(){ if [ -e "$1" ]; then echo "OK   $1"; else echo "MISS $1"; fi; }
sz(){ if [ -e "$1" ]; then echo "OK   $(du -sh "$1" 2>/dev/null | cut -f1)\t$1"; else echo "MISS $1"; fi; }

echo "============================================================"
echo " OrganoEnzymeGen HPC status   host=$(hostname)  user=${USER:-?}  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

hr; echo "[1] SCRATCH LAYOUT"
sz "$SCRATCH"
sz "$RFD2"
sz "$REPO"
sz "$SCRATCH/boltz-venv"
sz "$SCRATCH/lmpnn-venv"
sz "$SCRATCH/.boltz"            # Boltz weights + CCD cache
sz "$SCRATCH/LigandMPNN"

hr; echo "[2] GENERATOR (RFD2) READINESS"
yn "$SIF"
if [ -e "$SANDBOX/bin/sh" ]; then
  echo "OK   sandbox usable ($SANDBOX/bin/sh present)"
else
  echo "BAD  sandbox missing/corrupt: $SANDBOX/bin/sh  -> re-extract: bash cluster/extract_sif.sh $SIF"
fi
if [ -e "$WEIGHTS" ]; then echo "OK   RFD_173.pt  $(du -sh "$WEIGHTS" 2>/dev/null | cut -f1)"; else echo "MISS $WEIGHTS"; fi

hr; echo "[3] COMPILED MOTIFS (in repo, must be git-pulled)"
if [ -d "$REPO/pipeline/compiled" ]; then
  for d in "$REPO"/pipeline/compiled/*/; do
    t="$(basename "$d")"
    if [ -f "$d/motif.pdb" ] && [ -f "$d/manifest.json" ]; then echo "OK   $t"; else echo "INCOMPLETE $t"; fi
  done
else
  echo "MISS $REPO/pipeline/compiled  -> git pull in $REPO"
fi

hr; echo "[4] GENERATION PROGRESS  (designs per target in $ARM)"
printf "  %-30s %8s   %s\n" "TARGET" "DESIGNS" "scores.json / scores_sc.json"
for t in "${TARGETS[@]}"; do
  od="$ARM/$t"
  n=0; [ -d "$od" ] && n=$(find "$od" -maxdepth 1 -name '*-atomized-bb-False.pdb' 2>/dev/null | wc -l | tr -d ' ')
  l1="-"; [ -f "$od/scores.json" ] && l1="scores.json"
  l2="-"; [ -f "$SC/$t/scores_sc.json" ] && l2="scores_sc.json"
  printf "  %-30s %8s   L1:%-12s L2:%s\n" "$t" "$n" "$l1" "$l2"
done

hr; echo "[5] SCORE SUMMARIES (if present)"
for t in "${TARGETS[@]}"; do
  for f in "$ARM/$t/scores.json" "$SC/$t/scores_sc.json"; do
    [ -f "$f" ] || continue
    echo ">> $f"
    python3 - "$f" <<'PY' 2>/dev/null || echo "   (could not parse)"
import json,sys
d=json.load(open(sys.argv[1]))
if isinstance(d,dict):
    s=d.get("summary", {k:d[k] for k in d if not isinstance(d[k],(list,dict))})
    print("   summary:", json.dumps(s)[:600])
    for key in ("designs","results","per_design"):
        if isinstance(d.get(key),list):
            print(f"   n_{key}={len(d[key])}")
PY
  done
done

hr; echo "[6] SLURM — your jobs running/pending"
if command -v squeue >/dev/null 2>&1; then
  squeue -u "${USER:-$(whoami)}" -o "%.16i %.10P %.12j %.2t %.10M %.5D %R" 2>/dev/null || echo "  (squeue failed)"
else echo "  squeue not found (not on a SLURM node?)"; fi

hr; echo "[7] SLURM — your jobs since yesterday (sacct)"
if command -v sacct >/dev/null 2>&1; then
  sacct -X --starttime "$(date -d 'yesterday' '+%Y-%m-%d' 2>/dev/null || echo 1970-01-01)" \
    --format=JobID,JobName%18,Partition,State,Elapsed,ExitCode 2>/dev/null | head -40 || echo "  (sacct failed)"
else echo "  sacct not found"; fi

hr; echo "[8] GPU AVAILABILITY  (idle nodes/state on dgxlo + gpu)"
if command -v sinfo >/dev/null 2>&1; then
  sinfo -p dgxlo,gpu,beta -o "%.12P %.6a %.6D %.6t %.20G" 2>/dev/null | head -30 || echo "  (sinfo failed)"
else echo "  sinfo not found"; fi

hr; echo "[9] RECENT JOB LOGS (newest arm_arr_*.out / selfconsist / slurm-*.out)"
logs="$(find "$REPO" "$SCRATCH" "${HOME:-/nonexistent}" -maxdepth 2 \
        \( -name 'arm_arr_*.out' -o -name 'selfconsist*_*.out' -o -name 'slurm-*.out' \) \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -8 | cut -d' ' -f2-)"
if [ -n "$logs" ]; then
  echo "$logs"
  newest="$(printf '%s\n' "$logs" | head -1)"
  echo; echo ">> tail of newest: $newest"
  tail -n 25 "$newest" 2>/dev/null
else
  echo "  (no job logs found under $REPO / $SCRATCH / \$HOME)"
fi

hr; echo "[10] REPO GIT STATE ($REPO)"
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" log -1 --format='  HEAD %h %ci  %s' 2>/dev/null
  git -C "$REPO" status -sb 2>/dev/null | head -8
else echo "  $REPO is not a git repo"; fi

hr; echo "Done. (read-only — nothing was submitted)"
