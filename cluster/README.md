# cluster/ — Caltech HPC (SLURM) GPU reconnaissance

Three scripts to find out, fast and non-destructively, **what GPUs you can get and how soon** on the
Caltech HPC cluster — before committing to any architecture or long run.

| File | What it does | Submits a job? |
|---|---|---|
| `cluster_recon.sh` | Full survey: your limits, partitions, GPU inventory, **GPUs free right now**, queue pressure, and a **non-intrusive wait estimate** via `sbatch --test-only`. | No (unless `--probe`) |
| `gpu_probe.sbatch` | Tiny 5-min, 1-GPU job that prints the allocated GPU's model/VRAM/driver/CUDA. | It IS a job |
| `run_probe_and_time.sh` | Submits `gpu_probe.sbatch` and **measures the real submit→start wait**. | Yes (one tiny job) |

## Workflow (Windows dev → GitHub → cluster)

**1. Push from your machine** (these files committed to `github.com/ansschh/rfdiffusion`):
```powershell
# from your local rfdiffusion repo (copy this cluster/ folder into it first)
git add cluster/ ; git commit -m "Add SLURM GPU recon toolkit" ; git push
```

**2. On the Caltech HPC login node:**
```bash
git clone https://github.com/ansschh/rfdiffusion.git   # or: cd rfdiffusion && git pull
cd rfdiffusion/cluster
chmod +x *.sh                     # make executable
# If you see a '\r' / 'bad interpreter' error (CRLF from Windows), run once:
#   sed -i 's/\r$//' *.sh *.sbatch     (the .gitattributes should prevent this)
```

**3. Run the read-only survey first (no jobs queued):**
```bash
bash cluster_recon.sh
# tune the hypothetical request the wait-estimate uses:
REQ_GPUS=1 REQ_TIME=02:00:00 REQ_CPUS=8 REQ_MEM=32G bash cluster_recon.sh
```
Read the report it saves (`cluster_recon_<host>_<time>.txt`) in this order:
- **Section 1** = your ceiling (max GPUs / walltime / QOS you're allowed).
- **Section 4** = GPUs free *right now* (lines marked `FREE NOW` → immediate run possible).
- **Section 6** = predicted wait for your request, per GPU partition.

**4. Get the ground-truth wait (submits one tiny real job):**
```bash
bash cluster_recon.sh --probe
# or directly:
bash run_probe_and_time.sh
PROBE_PARTITION=gpu bash run_probe_and_time.sh     # force a specific partition
```
It prints `MEASURED submit->start wait: N seconds` and writes the allocated GPU's specs to
`gpuprobe_<jobid>.out`.

## Why this design
- **Nothing is hardcoded about Caltech's cluster** — partitions, GPU types, and limits are all
  *discovered* from `sinfo`/`scontrol`/`sacctmgr`, so the report reflects your real config, not an
  assumption.
- `sbatch --test-only` and `squeue --start` give SLURM's backfill *predictions* (they shift as the
  queue changes); "free GPUs right now" (Section 4) is the better signal for an immediate run; the
  `--probe` measurement is the truth.
- Read-only by default; the only thing that ever queues a job is `--probe` / `run_probe_and_time.sh`,
  and that job is 1 GPU for ≤5 minutes.

## RFdiffusion2 setup (after recon)

Confirmed cluster state (recon, 2026-05-25): Apptainer `1.3.3` module, 405 TB free scratch, instant
V100-32GB on `dgxlo`, login-node internet OK. RFD2 ships as an Apptainer container, so deps are
containerized — no conda hell.

```bash
# On a LOGIN NODE (needs internet), in scratch, inside tmux:
tmux new -s rfd2
bash /resnick/scratch/atiwari2/rfdiffusion/cluster/setup_rfd2.sh   # clone + download weights/container (>30 min)

# Then submit the smoke test to an instant V100:
sbatch /resnick/scratch/atiwari2/rfdiffusion/cluster/smoke_rfd2.sbatch
cat rfd2_smoke_*.out      # success = backbone .pdb files under pipeline_outputs/
```

Notes:
- RFD2 generation + LigandMPNN run fine on the V100. The repo's default **Chai-1** folding step likely
  errors on V100 (compute 7.0) — expected; we replace it with **AF2/AF3** for self-consistency. The
  smoke test only needs the *generation* stage to succeed.
- Everything lives in `/resnick/scratch` (home is over quota). `-B /resnick` binds scratch into the container.

## Next
Pick the validation folder (AF3 vs open-weight) — gates only the post-generation self-consistency step.
Then build the motif compiler: validated Tier-A trajectory (`audit/subQB_curation/...`) -> RFD2 input.
