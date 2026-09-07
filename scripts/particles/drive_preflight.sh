#!/bin/bash
# Perseus-side driver for the AMD preflight sequence.  Emits one progress line per
# milestone on stdout; every line is a notification, so keep them few and specific.
#   1. wait for any in-flight Plummer job
#   2. pull the branch on AMD and rebuild
#   3. PF-A  t=0 diagnostic, 4 cycles, full N            (validates trk + all 3 ledgers)
#   4. PF-B  frozen-metric orbit test, full N, 1 P_1/2   } submitted together
#   5. PF-C  short fully coupled run, 100 M, + restart   } (2 separate nodes)
set -uo pipefail
H=hpcfund.amd.com
R=/work1/eliasmost/jiaxiwu/plummer_s01_20260906
S() { ssh -o BatchMode=yes -o ConnectTimeout=30 $H "$@" 2>/dev/null; }

waitjob() {  # $1 = jobid, $2 = label
  local j=$1 l=$2 i
  for i in $(seq 1 720); do
    S "squeue -h -j $j -o %T" | grep -q . || return 0
    sleep 30
  done
  echo "$l: TIMED OUT waiting for job $j"
  return 1
}

verdict() {  # $1 = logfile glob, $2 = label
  local L
  L=$(S "ls -1t $1 2>/dev/null | head -1")
  [ -n "$L" ] || { echo "$2: no log found"; return 1; }
  S "grep -aoE 'CASE (DONE|FAILED|INCOMPLETE) \([a-z_0-9]+\)[^\n]{0,60}|Memory access fault[^\.]{0,40}|### FATAL ERROR[^\n]{0,100}|zone-cycles/cpu_second = [0-9.e+]+|Terminating on [a-z ]+ limit' \"$L\" | sort -u | head -6"
}

# ---- 1. drain any in-flight Plummer job
for j in $(S "squeue -h -u jiaxiwu -o '%i %j' | awk '\$2 ~ /^(pf_|plm_|bis_|prod_)/ {print \$1}'"); do
  waitjob "$j" "drain"
done
echo "DRIVER: queue drained, rebuilding at branch HEAD"

# ---- 2. pull + rebuild
S "cd $R/src/athenak && git fetch -q origin && git reset -q --hard origin/project/Plummer-cluster"
head=$(S "git -C $R/src/athenak rev-parse --short HEAD")
bj=$(S "cd $R && sbatch scripts/amd_build.sbatch" | grep -oE '[0-9]{5,}' | head -1)
echo "DRIVER: rebuild job $bj at $head"
waitjob "$bj" "build" || exit 1
if [ "$(S "grep -ac 'AMD BUILD OK' $R/logs/plm_build.$bj.log")" != "1" ]; then
  echo "DRIVER: BUILD $bj FAILED at $head"
  S "grep -aE 'error:|CMake Error' $R/logs/plm_build.$bj.log | head -20"
  exit 1
fi
echo "DRIVER: build OK at $head"

# ---- 3. PF-A
S "cd $R && rm -rf runs/pf_t0"
aj=$(S "cd $R && sbatch -t 00:30:00 -J pf_t0 scripts/amd_run.sbatch pf_t0 pf_t0_plummer" | grep -oE '[0-9]{5,}' | head -1)
echo "DRIVER: PF-A t=0 diagnostic submitted as $aj"
waitjob "$aj" "PF-A"
echo "DRIVER: PF-A $aj result:"
verdict "$R/logs/pf_t0.*.log" "PF-A"

# ---- 4/5. PF-B and PF-C together
S "cd $R && rm -rf runs/pf_frozen runs/pf_live"
fj=$(S "cd $R && sbatch -t 08:00:00 -J pf_frozen scripts/amd_run.sbatch pf_frozen pf_frozen_plummer" | grep -oE '[0-9]{5,}' | head -1)
lj=$(S "cd $R && sbatch -t 04:00:00 -J pf_live scripts/amd_run.sbatch pf_live pf_live_plummer" | grep -oE '[0-9]{5,}' | head -1)
echo "DRIVER: PF-B frozen-metric orbit test = $fj, PF-C live preflight = $lj"
waitjob "$lj" "PF-C"
echo "DRIVER: PF-C $lj result:"
verdict "$R/logs/pf_live.*.log" "PF-C"
waitjob "$fj" "PF-B"
echo "DRIVER: PF-B $fj result:"
verdict "$R/logs/pf_frozen.*.log" "PF-B"
echo "DRIVER: PREFLIGHT SEQUENCE COMPLETE (build $head; jobs A=$aj B=$fj C=$lj)"
