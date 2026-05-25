#!/usr/bin/env bash
# extract_sif.sh — unpack an lz4-compressed Apptainer .sif into a plain directory.
# Apptainer's bundled unsquashfs on this cluster is gzip-only, so we (1) pull the
# squashfs partition out of the SIF, then (2) extract it with an lz4-capable unsquashfs —
# first the standalone one from the squashfs module, falling back to a conda-forge
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

SQFS="$APPTAINER_TMPDIR/$(basename "${SIF%.sif}").sqfs"

echo "== SIF partition layout =="
apptainer sif list "$SIF"
FSID=$(apptainer sif list "$SIF" | awk -F'|' 'tolower($0) ~ /squashfs/ && $1 ~ /[0-9]/ {gsub(/[^0-9]/,"",$1); print $1; exit}')
[ -z "${FSID:-}" ] && FSID=1
echo "== dumping squashfs partition (descriptor $FSID) -> $SQFS =="
apptainer sif dump "$FSID" "$SIF" > "$SQFS" || { echo "sif dump failed"; exit 1; }
ls -lh "$SQFS"

extracted_ok() { [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; }

echo "== attempt 1: standalone 'unsquashfs' from the squashfs module =="
rm -rf "$OUT"
unsquashfs -f -d "$OUT" "$SQFS" 2>&1 | tee "$APPTAINER_TMPDIR/unsq1.log" | tail -6 || true
if extracted_ok && ! grep -qiE 'lz4|unsupported|not supported|failed' "$APPTAINER_TMPDIR/unsq1.log"; then
  echo "OK: extracted with the module's unsquashfs -> $OUT"; echo "$OUT"; exit 0
fi

echo "== module unsquashfs couldn't do lz4; fetching conda-forge squashfs-tools (has lz4) via micromamba =="
MM="$SCRATCH/micromamba"
if [ ! -x "$MM/bin/micromamba" ]; then
  mkdir -p "$MM"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$MM" bin/micromamba \
    || { echo "micromamba download failed"; exit 1; }
fi
export MAMBA_ROOT_PREFIX="$SCRATCH/mamba"
if [ ! -x "$SCRATCH/sqenv/bin/unsquashfs" ]; then
  "$MM/bin/micromamba" create -y -p "$SCRATCH/sqenv" -c conda-forge squashfs-tools \
    || { echo "micromamba create failed"; exit 1; }
fi
echo "== attempt 2: conda-forge unsquashfs =="
rm -rf "$OUT"
"$SCRATCH/sqenv/bin/unsquashfs" -f -d "$OUT" "$SQFS" 2>&1 | tail -10
if extracted_ok; then
  echo "OK: extracted with conda-forge unsquashfs -> $OUT"; echo "$OUT"; exit 0
fi

echo "FAILED to extract $SIF — paste the output above."; exit 1
