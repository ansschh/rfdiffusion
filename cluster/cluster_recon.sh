#!/usr/bin/env bash
# =============================================================================
# cluster_recon.sh  —  Exhaustive SLURM / GPU reconnaissance.
#
# WHAT IT DOES (read-only by default; submits nothing unless you pass --probe):
#   Reports, for the cluster you're logged into:
#     - who you are and what you're ALLOWED to request (accounts, QOS, assoc limits, fairshare)
#     - every partition and its limits (which are GPU partitions, max walltime, allowed QOS)
#     - the GPU hardware inventory (types and totals)
#     - GPUs that are FREE RIGHT NOW (per node, per type, per partition)
#     - queue pressure (pending GPU demand ahead of you)
#     - an ESTIMATED WAIT to get a GPU, computed non-intrusively via `sbatch --test-only`
#       and `squeue --start` (no job is actually queued)
#   With --probe it additionally submits one tiny real GPU job and times submit->start.
#
# USAGE (run on a LOGIN NODE):
#   bash cluster_recon.sh                 # full read-only survey
#   bash cluster_recon.sh --probe         # survey + submit a real 5-min 1-GPU timing probe
#   REQ_GPUS=1 REQ_TIME=01:00:00 REQ_CPUS=4 REQ_MEM=16G bash cluster_recon.sh
#       ^ tune the hypothetical request used for the wait-time estimate (wait depends on the ask)
#
# OUTPUT: prints to screen AND saves to cluster_recon_<host>_<timestamp>.txt
# =============================================================================

# NB: deliberately NOT using `set -e` — many of these commands are optional and
# differ across SLURM versions; we want to continue past any that fail.
set -uo pipefail

# ---- tunables (the hypothetical job used to estimate wait time) -------------
REQ_GPUS="${REQ_GPUS:-1}"
REQ_TIME="${REQ_TIME:-01:00:00}"
REQ_CPUS="${REQ_CPUS:-4}"
REQ_MEM="${REQ_MEM:-16G}"
DO_PROBE=0
[[ "${1:-}" == "--probe" ]] && DO_PROBE=1

REPORT="cluster_recon_$(hostname -s 2>/dev/null || echo host)_$(date +%Y%m%d_%H%M%S).txt"
# tee everything to the report file
exec > >(tee "$REPORT") 2>&1

hr(){ printf '\n%s\n' "==================================================================="; }
sec(){ hr; printf '## %s\n' "$*"; hr; }
run(){ # run a labeled command, tolerate absence/failure
  local label="$1"; shift
  printf '\n----- %s -----\n$ %s\n' "$label" "$*"
  command -v "${1%% *}" >/dev/null 2>&1 || { echo "[command not found: ${1}]"; return 0; }
  "$@" 2>&1 || echo "[command failed or unsupported flag: $*]"
}
have(){ command -v "$1" >/dev/null 2>&1; }

echo "cluster_recon.sh  —  $(date)"
echo "report file: $REPORT"
echo "hypothetical request used for wait estimate: ${REQ_GPUS} GPU(s), time ${REQ_TIME}, ${REQ_CPUS} CPU, ${REQ_MEM} mem"

if ! have sinfo; then
  echo
  echo "!! SLURM commands (sinfo) not found on this host."
  echo "!! Are you on a login node with the slurm module loaded? Try: module load slurm  (or check 'which sbatch')."
  exit 1
fi

# =============================================================================
sec "0. Scheduler identity"
run "SLURM version"                 sinfo --version
run "Cluster name & key config"     bash -c "scontrol show config | grep -E 'ClusterName|SchedulerType|SelectType|SelectTypeParameters|PriorityType|PreemptType|PreemptMode|DefMemPerCPU|MaxArraySize|GresTypes' || true"
run "This login host"               bash -c "hostname -f; uptime"

# =============================================================================
sec "1. WHO YOU ARE / WHAT YOU MAY REQUEST  (your hard limits)"
run "User + groups"                 bash -c "id; echo; echo USER=$USER"
run "Your associations (account/partition/QOS + limits)" \
    sacctmgr -p show associations user="$USER" \
    format=Account,User,Partition,QOS,DefaultQOS,GrpTRES,MaxTRES,MaxTRESPerNode,MaxWall,MaxJobs,MaxSubmit,GrpTRESRunMins
run "QOS definitions (limits that cap a single user/job)" \
    sacctmgr -p show qos \
    format=Name,Priority,MaxWall,MaxTRES,MaxTRESPerUser,MaxTRESPerAccount,GrpTRES,MaxJobsPerUser,MaxSubmitJobsPerUser,Flags,PreemptMode
run "Fairshare / usage (affects your queue priority)" sshare -U -l
run "Your current jobs"             squeue -u "$USER" -o "%.12i %.12P %.10j %.8T %.10M %.10l %.6D %.20S %R"
run "Recent job history (last 2 days: wait + run pattern)" \
    sacct -u "$USER" --starttime="$(date -d '2 days ago' +%Y-%m-%d 2>/dev/null || echo 2024-01-01)" \
    --format=JobID,Partition,QOS,State,Submit,Start,Elapsed,AllocTRES%40 -P

# =============================================================================
sec "2. PARTITIONS & LIMITS  (which are GPU; max walltime; allowed QOS/accounts)"
run "Partition summary (state/time/nodes)" sinfo -s
run "Per-partition full limits" \
    sinfo -o "%P | avail=%a | maxnodes=%D | maxtime=%l | default=%g | nodes=%D | state=%T"
run "scontrol partition detail (MaxTime, AllowQos, AllowAccounts, TRES weights)" \
    bash -c "scontrol show partition | sed -e 's/ /\n   /g' | grep -E 'PartitionName|State|MaxTime|DefaultTime|MaxNodes|MaxCPUsPerNode|AllowQos|AllowGroups|AllowAccounts|TRES=|TRESBillingWeights|QoS=|OverSubscribe|PreemptMode' || scontrol show partition"

# =============================================================================
sec "3. GPU HARDWARE INVENTORY (types & totals across the cluster)"
run "GRES per partition (raw)"      sinfo -o "%P %.6D %.10G %T"
run "GPU types present"             bash -c "sinfo -h -o '%G' | tr ',' '\n' | grep -i gpu | sort | uniq -c | sort -rn || echo 'no gpu gres advertised'"

# =============================================================================
sec "4. GPUs FREE RIGHT NOW  (per node / type / partition)  <<< headline"
# scontrol oneliner is the most parse-stable source for total vs used GPUs per node.
python3 - "$@" <<'PYEOF'
import subprocess, re, collections, sys

def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except Exception as e:
        return ""

def gpu_total_from_gres(s):
    # parse "gpu:a100:4", "gpu:4", "gpu:a100:4(S:0-1)" -> sum of trailing ints
    if not s or s in ("(null)", "N/A"):
        return 0, {}
    s = re.sub(r'\(S:[^)]*\)', '', s)            # strip socket suffixes
    total = 0; bytype = collections.Counter()
    for m in re.finditer(r'gpu:(?:([^:,()]+):)?(\d+)', s):
        typ = m.group(1) or "gpu"
        n = int(m.group(2))
        total += n; bytype[typ] += n
    return total, dict(bytype)

raw = sh(["scontrol","show","nodes","--oneliner"])
if not raw.strip():
    print("[scontrol show nodes returned nothing]"); sys.exit(0)

rows = []
agg_free = collections.Counter()        # by gpu type
agg_total = collections.Counter()
part_free = collections.defaultdict(collections.Counter)   # partition -> type -> free
for line in raw.splitlines():
    d = dict(re.findall(r'(\w+)=(\S+)', line))
    name = d.get("NodeName","?")
    state = d.get("State","?")
    parts = d.get("Partitions","")
    tot, tot_by = gpu_total_from_gres(d.get("Gres",""))
    used, used_by = gpu_total_from_gres(d.get("GresUsed",""))
    # fallback to TRES if Gres/GresUsed absent
    if tot == 0:
        m = re.search(r'gres/gpu=(\d+)', d.get("CfgTRES",""));  tot = int(m.group(1)) if m else 0
    if used == 0:
        m = re.search(r'gres/gpu=(\d+)', d.get("AllocTRES",""));  used = int(m.group(1)) if m else 0
    if tot == 0:
        continue                                   # not a GPU node
    free = max(tot - used, 0)
    cpu_alloc = d.get("CPUAlloc","?"); cpu_tot = d.get("CPUTot","?")
    freemem = d.get("FreeMem","?"); realmem = d.get("RealMemory","?")
    usable = state.upper().startswith(("IDLE","MIX"))
    rows.append((name, state, parts, tot, used, free, f"{cpu_alloc}/{cpu_tot}", freemem, realmem, tot_by, usable))
    for t,n in (tot_by or {"gpu":tot}).items():
        agg_total[t]+=n
    if usable and free>0:
        # attribute free GPUs to this node's type(s)
        for t in (tot_by or {"gpu":tot}):
            agg_free[t]+=free
        for p in parts.split(","):
            for t in (tot_by or {"gpu":tot}):
                part_free[p][t]+=free

print(f"{'NODE':18} {'STATE':10} {'GPUtot':6} {'used':4} {'FREE':4} {'CPU a/t':9} {'FreeMem(MB)':11} {'types'}")
for r in sorted(rows, key=lambda x:(-x[5], x[0])):   # most-free first
    flag = " <-- FREE NOW" if (r[10] and r[5]>0) else ""
    print(f"{r[0][:18]:18} {r[1][:10]:10} {r[3]:6} {r[4]:4} {r[5]:4} {r[6]:9} {str(r[7]):11} {r[9]}{flag}")

print("\n--- AGGREGATE GPUs free right now (idle/mix nodes) by type ---")
if agg_free:
    for t,n in sorted(agg_free.items(), key=lambda x:-x[1]):
        print(f"   {t:14} free={n:4}   total={agg_total.get(t,'?')}")
else:
    print("   0 GPUs currently free on idle/mix nodes (everything allocated or down/drained).")

print("\n--- free GPUs by partition (idle/mix) ---")
for p in sorted(part_free):
    tot = sum(part_free[p].values())
    detail = ", ".join(f"{t}:{n}" for t,n in part_free[p].items())
    print(f"   {p:20} free={tot:4}  ({detail})")
PYEOF

# =============================================================================
sec "5. QUEUE PRESSURE  (pending GPU demand ahead of you)"
run "Job state counts (all)"        bash -c "squeue -h -o '%T' | sort | uniq -c | sort -rn"
run "Running GPU jobs (tres-per-job)" bash -c "squeue -t RUNNING -o '%.12i %.10P %.8u %.6D %.12b %.10M %.10L' | head -40"
run "PENDING jobs requesting GPUs (who's ahead of you)" \
    bash -c "squeue -t PENDING -o '%.12i %.10P %.8u %.8Q %.12b %.20S %r' | grep -iE 'gpu|gres' | head -60 || echo 'none pending for gpu'"
run "Pending-job priority ranking (where you'd land)" bash -c "sprio -l 2>/dev/null | head -40 || echo 'sprio unavailable'"

# =============================================================================
sec "6. WAIT-TIME ESTIMATE via sbatch --test-only  (NON-INTRUSIVE: queues nothing)"
echo "Estimating start time for a hypothetical job: --gres=gpu:${REQ_GPUS} --time=${REQ_TIME} --cpus-per-task=${REQ_CPUS} --mem=${REQ_MEM}"
# discover GPU partitions (those advertising gpu gres)
GPU_PARTS=$(sinfo -h -o "%P %G" | awk 'tolower($2) ~ /gpu/ {gsub(/\*/,"",$1); print $1}' | sort -u)
if [[ -z "${GPU_PARTS}" ]]; then
  echo "No GPU partitions auto-detected from sinfo GRES. Falling back to all partitions."
  GPU_PARTS=$(sinfo -h -o "%P" | sed 's/\*//' | sort -u)
fi
echo "GPU partitions detected: ${GPU_PARTS:-none}"
TMPJOB="$(mktemp /tmp/gpuprobe.XXXXXX.sbatch)"
cat > "$TMPJOB" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=ttonly
#SBATCH --gres=gpu:${REQ_GPUS}
#SBATCH --cpus-per-task=${REQ_CPUS}
#SBATCH --mem=${REQ_MEM}
#SBATCH --time=${REQ_TIME}
#SBATCH --output=/dev/null
echo placeholder
EOF
for P in ${GPU_PARTS}; do
  printf '\n----- partition: %s -----\n' "$P"
  echo "\$ sbatch --test-only --partition=$P $TMPJOB"
  sbatch --test-only --partition="$P" "$TMPJOB" 2>&1 \
    || echo "[--test-only failed for $P: you may lack access, or QOS/gres mismatch — see Section 1 for what you're allowed]"
  # squeue --start gives backfill estimates for already-pending jobs on this partition
  echo "\$ squeue --start -p $P -t PENDING (existing pending jobs' predicted starts)"
  squeue --start -p "$P" -t PENDING -o "%.12i %.8u %.12b %.20S %r" 2>&1 | head -15 || true
done
rm -f "$TMPJOB"
echo
echo "NOTE: --test-only and --start are SLURM BACKFILL PREDICTIONS — they assume the current"
echo "queue and shift as jobs come/go. 'free GPUs right now' in Section 4 is the better signal"
echo "for an IMMEDIATE run; the empirical --probe below is the ground truth."

# =============================================================================
if [[ "$DO_PROBE" -eq 1 ]]; then
  sec "7. EMPIRICAL PROBE  (submits ONE real 5-min 1-GPU job and times submit->start)"
  bash "$(dirname "$0")/run_probe_and_time.sh" || echo "[probe wrapper failed]"
else
  sec "7. EMPIRICAL PROBE — skipped"
  echo "Re-run with:  bash cluster_recon.sh --probe   to submit a real timing probe."
fi

# =============================================================================
sec "8. SUMMARY / WHAT TO DO NEXT"
cat <<'EOS'
Read the report in this order:
  * Section 1  -> the ceiling: max GPUs / walltime / QOS you are ALLOWED to request.
  * Section 4  -> GPUs free RIGHT NOW. If any line says "FREE NOW", you can likely start immediately
                  by targeting that partition with the matching --gres=gpu:<type>:N.
  * Section 6  -> predicted wait for your hypothetical request, per GPU partition.
  * Section 7  -> (with --probe) the real measured submit->start latency.

To launch an immediate experiment once you've picked a partition P and GPU count N:
  sbatch --partition=P --gres=gpu:N --time=HH:MM:SS --cpus-per-task=C --mem=XXG your_job.sbatch
EOS
echo
echo "Saved full report to: $REPORT"
