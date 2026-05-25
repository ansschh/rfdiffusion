#!/usr/bin/env bash
# setup_rfd2.sh — install RFdiffusion2 on Caltech HPC (Resnick / "central").
# RUN ON A LOGIN NODE (it needs internet) FROM SCRATCH. Downloads the Apptainer
# container + model weights into scratch — LARGE, can take >30 min.
#
#   TIP: run inside tmux/screen so it survives an SSH disconnect:
#     tmux new -s rfd2    (then run this; detach with Ctrl-b d, reattach: tmux attach -t rfd2)
#
# Env overrides: SCRATCH_DIR=... APPT_MODULE=...
set -uo pipefail

SCRATCH="${SCRATCH_DIR:-/resnick/scratch/atiwari2}"
APPT_MODULE="${APPT_MODULE:-apptainer/1.3.3-gcc-13.2.0-i5n6b74}"

echo "== loading apptainer module: $APPT_MODULE =="
module load "$APPT_MODULE" 2>/dev/null || module load apptainer 2>/dev/null || true
apptainer --version || { echo "ERROR: apptainer not available after 'module load'. Try: module avail apptainer"; exit 1; }

cd "$SCRATCH" || { echo "ERROR: cannot cd to $SCRATCH"; exit 1; }
if [ ! -d RFdiffusion2 ]; then
  echo "== cloning RFdiffusion2 into $SCRATCH =="
  git clone https://github.com/RosettaCommons/RFdiffusion2.git
fi
cd RFdiffusion2
export PYTHONPATH="$PWD"
echo "PYTHONPATH=$PYTHONPATH"

echo "== downloading weights + container via setup.py (LARGE; >30 min) =="
if ! python3 setup.py; then
  echo
  echo "setup.py failed. Common causes & fixes:"
  echo "  * missing python module -> python3 -m pip install --user <module>, then re-run:"
  echo "        python3 setup.py overwrite"
  echo "  * download interrupted   -> resume with: python3 setup.py overwrite"
  exit 1
fi

echo "== verifying the container landed =="
SIF="rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif"
if [ -f "$SIF" ]; then
  ls -lh "$SIF"
  echo
  echo "RFD2 setup looks complete."
  echo "Next: submit the smoke test ->  sbatch /resnick/scratch/atiwari2/rfdiffusion/cluster/smoke_rfd2.sbatch"
else
  echo "WARNING: $SIF not found. Inspect setup.py output above; you may need: python3 setup.py overwrite"
  exit 1
fi
