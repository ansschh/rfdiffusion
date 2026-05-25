#!/usr/bin/env bash
# extract_sif.sh — unpack an lz4-compressed Apptainer .sif into a plain directory.
# The cluster's apptainer/unsquashfs is gzip-only, so we read the squashfs partition
# STRAIGHT OUT OF THE SIF at its byte offset (unsquashfs -o) with an lz4-capable
# unsquashfs — the squashfs module's tool first, falling back to conda-forge
# squashfs-tools fetched via micromamba into scratch (no admin needed).
#
# Usage:  bash extract_sif.sh <path-to .sif> [output_dir]
# Result: a runnable container directory you exec with:  apptainer exec --nv -B /resnick <dir> ...
set -uo pipefail

SIF="${1:?usage: extract_sif.sh <sif> [outdir]}"
OUT="${2:-${SIF%.sif}_sandbox}"
SCRATCH="${SCRATCH_DIR:-/resnick/scratch/atiwari2}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH/.apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$SCRATCH/apptmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
module load apptainer/1.3.3-gcc-13.2.0-i5n6b74 2>/dev/null || module load apptainer 2>/dev/null || true

echo "== SIF partition layout =="
apptainer sif list "$SIF"
OFFSET=$(apptainer sif list "$SIF" | awk -F'|' 'tolower($0) ~ /squashfs/ {split($4,a,"-"); gsub(/[^0-9]/,"",a[1]); print a[1]; exit}')
[ -z "${OFFSET:-}" ] && { echo "could not detect squashfs offset from sif list"; exit 1; }
echo "== squashfs partition offset in SIF: $OFFSET =="

extracted_ok() { [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; }

echo "== attempt 1: standalone 'unsquashfs' from the squashfs module (reads SIF at offset) =="
rm -rf "$OUT"
unsquashfs -o "$OFFSET" -f -d "$OUT" "$SIF" 2>&1 | tee "$APPTAINER_TMPDIR/unsq1.log" | tail -6 || true
if extracted_ok && ! grep -qiE 'lz4|unsupported|not supported|EOF|failed' "$APPTAINER_TMPDIR/unsq1.log"; then
  echo "OK: extracted with the module's unsquashfs -> $OUT"; exit 0
fi

echo "== module unsquashfs can't do lz4; fetching conda-forge squashfs-tools (has lz4) via micromamba =="
MM="$SCRATCH/micromamba"
if [ ! -x "$MM/bin/micromamba" ]; then
  mkdir -p "$MM"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$MM" bin/micromamba \
    || { echo "micromamba download failed"; exit 1; }
fi
export MAMBA_ROOT_PREFIX="$SCRATCH/mamba"
CONDA_HOME="$SCRATCH/condahome"; mkdir -p "$CONDA_HOME"   # keep conda's ~/.conda off the over-quota home
if [ ! -x "$SCRATCH/sqenv/bin/unsquashfs" ]; then
  HOME="$CONDA_HOME" "$MM/bin/micromamba" create -y -p "$SCRATCH/sqenv" -c conda-forge squashfs-tools || true
fi
[ -x "$SCRATCH/sqenv/bin/unsquashfs" ] || { echo "could not obtain conda-forge unsquashfs"; exit 1; }

echo "== attempt 2: conda-forge unsquashfs (reads SIF at offset) =="
rm -rf "$OUT"
"$SCRATCH/sqenv/bin/unsquashfs" -o "$OFFSET" -f -d "$OUT" "$SIF" 2>&1 | tail -10
if extracted_ok; then
  echo "OK: extracted with conda-forge unsquashfs -> $OUT"; exit 0
fi
echo "FAILED to extract $SIF — paste the output above."; exit 1
