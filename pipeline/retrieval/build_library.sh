#!/usr/bin/env bash
# build_library.sh — fetch + extract pockets for the Tier-1 seed library.
# Reads seed_library.txt (PDB_ID metal_element note), fetches each PDB from RCSB
# (auto via extract_pocket), extracts the typed pocket, skips failures with a warning.
#
# Usage:
#   bash pipeline/retrieval/build_library.sh
#   bash pipeline/retrieval/build_library.sh path/to/seed.txt  path/to/output_dir
set -uo pipefail

SEED="${1:-pipeline/retrieval/seed_library.txt}"
OUT="${2:-pipeline/retrieval/library}"
mkdir -p "$OUT/pdb" "$OUT/pockets"

ok=0; fail=0
while IFS= read -r line || [ -n "$line" ]; do
  # skip blank lines and comments
  line="${line#"${line%%[![:space:]]*}"}"     # ltrim
  [ -z "$line" ] && continue
  case "$line" in \#*) continue ;; esac

  pdb="$(echo "$line" | awk '{print $1}')"
  metal="$(echo "$line" | awk '{print $2}')"
  [ -z "$pdb" ] || [ -z "$metal" ] && continue

  pdb_path="$OUT/pdb/${pdb}.pdb"
  out_json="$OUT/pockets/${pdb}_${metal}.json"

  echo "--- $pdb / $metal ---"
  if python pipeline/retrieval/extract_pocket.py "$pdb_path" --metal "$metal" --out "$out_json" 2>&1 | tail -8; then
    if [ -f "$out_json" ]; then
      ok=$((ok+1))
    else
      fail=$((fail+1)); echo "  SKIP $pdb ($metal): no pocket output"
    fi
  else
    fail=$((fail+1)); echo "  SKIP $pdb ($metal): extraction error"
  fi
done < "$SEED"

echo
echo "=== library built: $ok pockets ok, $fail skipped ==="
echo "  pockets in: $OUT/pockets"
ls "$OUT/pockets" 2>/dev/null | sed 's/^/    /'
