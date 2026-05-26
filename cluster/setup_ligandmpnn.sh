#!/usr/bin/env bash
# setup_ligandmpnn.sh — install LigandMPNN (sequence design on RFD2 backbones) into scratch.
# Layer-2 step 1. RUN ON A LOGIN NODE (needs internet). Reuses the Boltz venv (torch+cuda).
# LigandMPNN is atom-aware: it ingests the Cp*Ir cofactor as ligand context directly
# (no CCD/SMILES needed), which is exactly why it fits the organometallic case.
#   tip: run inside tmux.
set -uo pipefail

SCRATCH="${SCRATCH_DIR:-/resnick/scratch/atiwari2}"
LMPNN="$SCRATCH/LigandMPNN"
VENV="${BOLTZ_VENV:-$SCRATCH/boltz-venv}"
export HOME="${SCRATCH}/condahome"   # keep pip/caches off the over-quota home
mkdir -p "$HOME"

module load python/3.11.6 2>/dev/null || module load python/3.11 2>/dev/null || module load python3 2>/dev/null || true

echo "== clone LigandMPNN =="
[ -d "$LMPNN/.git" ] || git clone https://github.com/dauparas/LigandMPNN.git "$LMPNN"
cd "$LMPNN"

echo "== activate Boltz venv ($VENV) — reuses its torch+cuda =="
source "$VENV/bin/activate" || { echo "ERROR: $VENV missing — run setup_boltz.sh first"; exit 1; }

echo "== install LigandMPNN python deps (torch already present from boltz) =="
pip install -q numpy ml_collections 2>/dev/null || true
[ -f requirements.txt ] && pip install -q -r requirements.txt 2>/dev/null || true

echo "== download model weights (-> $LMPNN/model_params) =="
if [ ! -d model_params ] || [ -z "$(ls -A model_params 2>/dev/null)" ]; then
  bash get_model_params.sh "./model_params" || { echo "weight download failed — check get_model_params.sh"; exit 1; }
fi
echo "== model_params =="
ls -lh model_params 2>/dev/null | head

echo "== sanity: run.py usage =="
python run.py --help 2>&1 | head -30 || echo "(run.py --help failed; check args in the next step)"
echo
echo "Done. LigandMPNN=$LMPNN   venv=$VENV"
echo "Next: sbatch cluster/selfconsist_smoke.sbatch 3ZP9"
