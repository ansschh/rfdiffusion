#!/usr/bin/env bash
# setup_ligandmpnn.sh — install LigandMPNN in ITS OWN venv, and repair the Boltz venv.
# LigandMPNN (ProDy) needs numpy<1.24; Boltz (numba) needs numpy>=1.24 — incompatible, so
# they MUST be separate venvs. Layer-2 runs them as separate steps. RUN ON A LOGIN NODE
# (internet); use tmux (torch install is large).
set -uo pipefail

SCRATCH="${SCRATCH_DIR:-/resnick/scratch/atiwari2}"
LMPNN="$SCRATCH/LigandMPNN"
LMPNN_VENV="${LMPNN_VENV:-$SCRATCH/lmpnn-venv}"
BOLTZ_VENV="${BOLTZ_VENV:-$SCRATCH/boltz-venv}"
export HOME="${SCRATCH}/condahome"; mkdir -p "$HOME"
module load python/3.11.6 2>/dev/null || module load python/3.11 2>/dev/null || module load python3 2>/dev/null || true
PY="$(command -v python3.11 || command -v python3)"

echo "== clone LigandMPNN =="
[ -d "$LMPNN/.git" ] || git clone https://github.com/dauparas/LigandMPNN.git "$LMPNN"

echo "== LigandMPNN venv ($LMPNN_VENV) — separate from Boltz =="
[ -d "$LMPNN_VENV" ] || "$PY" -m venv "$LMPNN_VENV"
source "$LMPNN_VENV/bin/activate"
python -m pip install -q --upgrade pip wheel
echo "   installing torch (cuda) + LigandMPNN runtime deps (numpy<1.24 for ProDy)..."
pip install -q torch                       # default wheel bundles CUDA (works on the V100)
pip install -q 'numpy<1.24' prody ml_collections scipy
cd "$LMPNN"
[ -d model_params ] && [ -n "$(ls -A model_params 2>/dev/null)" ] || bash get_model_params.sh "./model_params"
echo "   LigandMPNN sanity:"; python run.py --help 2>&1 | head -3 || echo "(run.py --help failed)"
deactivate

echo "== repair Boltz venv ($BOLTZ_VENV) — the earlier shared install downgraded its deps =="
source "$BOLTZ_VENV/bin/activate"
pip install -q 'boltz==2.2.1'              # repins numpy>=1.24, biopython==1.84, scipy==1.13.1
python -c "import boltz, numba, numpy; print('   boltz venv OK; numpy', numpy.__version__)" \
  || echo '!! boltz venv still broken — paste this output'
deactivate

echo
echo "Done. LigandMPNN venv=$LMPNN_VENV   Boltz venv=$BOLTZ_VENV   weights=$LMPNN/model_params"
echo "Next: sbatch cluster/selfconsist_smoke.sbatch 3ZP9"
