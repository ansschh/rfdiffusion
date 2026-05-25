#!/usr/bin/env bash
# setup_boltz.sh — install Boltz-2 and pre-download its weights into SCRATCH.
# RUN ON A LOGIN NODE (needs internet). Boltz needs python >= 3.10.
# Weights go to $BOLTZ_CACHE in scratch (home is over quota; compute nodes are offline,
# so the weights MUST be fetched here once).
#   tip: run inside tmux so the pip install + download survive disconnects.
set -uo pipefail

SCRATCH="${SCRATCH_DIR:-/resnick/scratch/atiwari2}"
VENV="${BOLTZ_VENV:-$SCRATCH/boltz-venv}"
export BOLTZ_CACHE="${BOLTZ_CACHE:-$SCRATCH/.boltz}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$BOLTZ_CACHE"

echo "== available python modules (need >= 3.10) =="
module avail python 2>&1 | tail -20 || true
echo "== attempting to load a python >= 3.10 module =="
module load python/3.11.6 2>/dev/null || module load python/3.11 2>/dev/null \
  || module load python/3.10 2>/dev/null || module load python3 2>/dev/null || true
PY="$(command -v python3.11 || command -v python3.10 || command -v python3)"
echo "using interpreter: $PY ($("$PY" --version 2>&1))"
if [ "$("$PY" -c 'import sys;print(int(sys.version_info[:2]>=(3,10)))' 2>/dev/null)" != "1" ]; then
  echo "!! Boltz needs python >= 3.10 but found $("$PY" --version 2>&1)."
  echo "!! Run 'module avail python', load a >=3.10 module, then re-run this script."
  echo "!! (continuing anyway in case the venv python differs)"
fi

echo "== creating venv at $VENV =="
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

echo "== pip install boltz[cuda]  (pulls torch+cuda; several minutes) =="
pip install -U "boltz[cuda]" || { echo "pip install failed — paste the error"; exit 1; }

echo "== pre-downloading Boltz-2 weights into $BOLTZ_CACHE (CPU fold of a 20-aa peptide) =="
boltz predict "$HERE/boltz_prefetch.yaml" \
  --cache "$BOLTZ_CACHE" --out_dir "$SCRATCH/boltz_prefetch_out" \
  --accelerator cpu --devices 1 --no_kernels --override \
  || echo "(prefetch inference errored — fine as long as weights downloaded; check $BOLTZ_CACHE)"

echo "== BOLTZ_CACHE contents =="
ls -lh "$BOLTZ_CACHE" 2>/dev/null | head
echo
echo "Done.  venv=$VENV   BOLTZ_CACHE=$BOLTZ_CACHE"
echo "Next:  sbatch /resnick/scratch/atiwari2/rfdiffusion/cluster/boltz_smoke.sbatch"
