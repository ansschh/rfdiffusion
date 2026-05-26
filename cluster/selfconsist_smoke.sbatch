#!/usr/bin/env bash
# selfconsist_smoke.sbatch — layer-2 mechanism smoke on ONE RFD2 design:
#   LigandMPNN (sequence design, cofactor as atom context) -> Boltz-2 single-seq fold.
# Proves the design->fold loop runs before we scale to top-K + scoring. Same incremental
# de-risking we used for RFD2.
#   sbatch cluster/selfconsist_smoke.sbatch 3ZP9
#SBATCH --job-name=sc_smoke
#SBATCH --partition=dgxlo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=sc_smoke_%j.out
set -uo pipefail

TARGET="${1:-3ZP9}"
SCRATCH="${SCRATCH_DIR:-/resnick/scratch/atiwari2}"
LMPNN="$SCRATCH/LigandMPNN"
VENV="${BOLTZ_VENV:-$SCRATCH/boltz-venv}"
export BOLTZ_CACHE="${BOLTZ_CACHE:-$SCRATCH/.boltz}"
export HOME="${SCRATCH}/condahome"; mkdir -p "$HOME"
module load python/3.11.6 2>/dev/null || module load python3 2>/dev/null || true
source "$VENV/bin/activate" || { echo "ERROR: boltz venv missing"; exit 1; }

DESIGNDIR="$SCRATCH/RFdiffusion2/arm_designs/$TARGET"
DESIGN="$(find "$DESIGNDIR" -maxdepth 1 -name '*cond0_0-atomized-bb-False.pdb' 2>/dev/null | head -1)"
[ -f "$DESIGN" ] || { echo "ERROR: no design PDB for $TARGET in $DESIGNDIR"; exit 1; }
WORK="$SCRATCH/selfconsist/$TARGET"; mkdir -p "$WORK/lmpnn"
echo "=== $(hostname) | design: $DESIGN ==="
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# Strip the ORI centering token (a fake pseudo-atom, not a real ligand) before MPNN/Boltz.
grep -v ' ORI ' "$DESIGN" > "$WORK/design.pdb"

echo "=== (1) LigandMPNN sequence design (cofactor LIG as atom context) ==="
CKPT="$(ls "$LMPNN"/model_params/ligandmpnn_v_*.pt 2>/dev/null | head -1)"
[ -f "$CKPT" ] || { echo "ERROR: no LigandMPNN weights — run setup_ligandmpnn.sh"; exit 1; }
cd "$LMPNN"
python run.py --model_type ligand_mpnn --checkpoint_ligand_mpnn "$CKPT" \
  --pdb_path "$WORK/design.pdb" --out_folder "$WORK/lmpnn" \
  --number_of_batches 4 --batch_size 1 --temperature 0.1
echo "=== LigandMPNN exit $? ==="
ls -R "$WORK/lmpnn" | head -30

echo "=== (2) extract first designed sequence ==="
SEQ="$(python3 -c "
import glob
fas=glob.glob('$WORK/lmpnn/seqs/*.fa') or glob.glob('$WORK/lmpnn/seqs/*.fasta')
recs=open(fas[0]).read().split('>')[1:]
rec=recs[1] if len(recs)>1 else recs[0]          # [0]=input native, [1]=first design
seq=''.join(c for c in rec.split(chr(10),1)[1] if c.isalpha())
print(seq)
")"
echo "designed sequence (${#SEQ} aa): $SEQ"
[ -n "$SEQ" ] || { echo 'ERROR: could not parse a designed sequence'; exit 1; }

echo "=== (3) Boltz-2 single-sequence fold (protein self-consistency, no cofactor) ==="
cat > "$WORK/fold.yaml" <<EOF
sequences:
  - protein:
      id: A
      sequence: $SEQ
      msa: empty
EOF
boltz predict "$WORK/fold.yaml" --cache "$BOLTZ_CACHE" --out_dir "$WORK/boltz_out" \
  --accelerator gpu --devices 1 --no_kernels --override
echo "=== boltz exit $? ==="
find "$WORK/boltz_out" -name '*.cif' -o -name '*.json' 2>/dev/null | head
echo
echo "SUCCESS = a predicted .cif under $WORK/boltz_out. Then score_selfconsistency.py"
echo "compares it to the RFD2 backbone (CA-RMSD + pLDDT) = layer-2 self-consistency."
