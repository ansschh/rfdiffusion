#!/usr/bin/env bash
# selfconsist.sbatch — layer-2 self-consistency over the TOP-K designs of a target (Rev2).
# For each of the top-K RFD2 designs (by layer-1 buried_core_frac): LigandMPNN designs NSEQ
# sequences -> one Boltz-2 fold over ALL of them (single model load) -> best-of-N CA-RMSD +
# pLDDT scoring. Needs scores.json (run score_designs.py on the target first).
# Boltz needs bf16 -> Ampere+ (L40S/H100/H200 on gpu/beta).
#   sbatch cluster/selfconsist.sbatch 3ZP9
#   TOPK=5 NSEQ=4 sbatch cluster/selfconsist.sbatch 3ZP9
#SBATCH --job-name=selfcon
#SBATCH --partition=gpu,beta
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=selfcon_%j.out
set -uo pipefail

TARGET="${1:-3ZP9}"; TOPK="${TOPK:-5}"; NSEQ="${NSEQ:-4}"
SCRATCH="${SCRATCH_DIR:-/resnick/scratch/atiwari2}"
REPO="${REPO_DIR:-/resnick/scratch/atiwari2/rfdiffusion}"
LMPNN="$SCRATCH/LigandMPNN"
LMPNN_VENV="${LMPNN_VENV:-$SCRATCH/lmpnn-venv}"
BOLTZ_VENV="${BOLTZ_VENV:-$SCRATCH/boltz-venv}"
export BOLTZ_CACHE="${BOLTZ_CACHE:-$SCRATCH/.boltz}"
module load python/3.11.6 2>/dev/null || module load python3 2>/dev/null || true

DESIGNDIR="$SCRATCH/RFdiffusion2/arm_designs/$TARGET"
SCORES="$DESIGNDIR/scores.json"
[ -f "$SCORES" ] || { echo "ERROR: $SCORES missing — run score_designs.py on $TARGET first"; exit 1; }
WORK="$SCRATCH/selfconsist/$TARGET"
rm -rf "$WORK/inputs" "$WORK/folds"; mkdir -p "$WORK/designs" "$WORK/inputs" "$WORK/folds" "$WORK/lmpnn"
CKPT="$(ls "$LMPNN"/model_params/ligandmpnn_v_*.pt 2>/dev/null | head -1)"

TOP="$(python3 -c "
import json
d=[x for x in json.load(open('$SCORES'))['designs'] if x.get('metal_present')]
d.sort(key=lambda x:x['buried_core_frac'], reverse=True)
print('\n'.join(x['file'] for x in d[:$TOPK]))
")"
echo "=== $(hostname) | TARGET=$TARGET TOPK=$TOPK NSEQ=$NSEQ ==="
echo "top-K designs:"; echo "$TOP"

echo "=== (1) LigandMPNN: $NSEQ sequences per top-K design ==="
source "$LMPNN_VENV/bin/activate"
for f in $TOP; do
  d="${f%.pdb}"
  grep -v ' ORI ' "$DESIGNDIR/$f" > "$WORK/designs/$d.pdb"
  cd "$LMPNN"
  python run.py --model_type ligand_mpnn --checkpoint_ligand_mpnn "$CKPT" \
    --pdb_path "$WORK/designs/$d.pdb" --out_folder "$WORK/lmpnn/$d" \
    --number_of_batches "$NSEQ" --batch_size 1 --temperature 0.1 || { echo "lmpnn failed for $d"; continue; }
  python3 - "$WORK/lmpnn/$d" "$d" "$WORK/inputs" <<'PY'
import sys, glob, os
lmdir, did, inp = sys.argv[1:4]
fa = (glob.glob(os.path.join(lmdir, "seqs", "*.fa")) or glob.glob(os.path.join(lmdir, "seqs", "*.fasta")))[0]
recs = open(fa).read().split(">")[1:]
seqs = [''.join(c for c in r.split(chr(10), 1)[1] if c.isalpha()) for r in recs[1:]]  # [0]=native
for i, s in enumerate(seqs):
    if s:
        open(os.path.join(inp, f"{did}__s{i}.yaml"), "w").write(
            "sequences:\n  - protein:\n      id: A\n      sequence: %s\n      msa: empty\n" % s)
print(f"  {did}: wrote {sum(1 for s in seqs if s)} seqs")
PY
done
deactivate

echo "=== (2) Boltz-2 fold over ALL sequences (one model load) ==="
N_IN=$(ls "$WORK/inputs"/*.yaml 2>/dev/null | wc -l)
[ "$N_IN" -gt 0 ] || { echo "ERROR: no input yamls written"; exit 1; }
echo "    folding $N_IN sequences..."
source "$BOLTZ_VENV/bin/activate"
boltz predict "$WORK/inputs" --cache "$BOLTZ_CACHE" --out_dir "$WORK/folds" \
  --accelerator gpu --devices 1 --no_kernels --override
echo "=== boltz exit $? ==="

echo "=== (3) best-of-N self-consistency scoring ==="
python "$REPO/pipeline/score_selfconsistency.py" --target-dir "$WORK"
deactivate
echo
echo "wrote $WORK/scores_sc.json  (per-design best-of-$NSEQ CA-RMSD + pLDDT, top-$TOPK aggregate)"
