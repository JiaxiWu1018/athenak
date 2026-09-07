# Storage watchdog (campaign convention, preserved verbatim from
# reconstruction_ST/04c_ST_AH_particle_excision_20260821/scripts/amd_watchdog.inc).
# Two non-negotiable details: the subshell must run `set +e +o pipefail` so a transient
# `du` failure cannot kill it, and it must poll the WHOLE user root, not just this
# campaign, because other campaigns share the 1.9 TiB budget.
start_watchdog () {
  UROOT=/work1/eliasmost/jiaxiwu
  local croot=$1
  WARN_KB=1610612736   # 1.5 TiB
  KILL_KB=1825361100   # 1.7 TiB
  HARD_KB=2040109465   # 1.9 TiB
  ( set +e +o pipefail
    prev=0
    while true; do
      u=$(du -sk ${UROOT} 2>/dev/null | awk '{print $1}')
      c=$(du -sk ${croot} 2>/dev/null | awk '{print $1}')
      [ -n "${u}" ] || u=${prev}; [ -n "${c}" ] || c=0
      g=0; [ "${prev}" -gt 0 ] && g=$((u - prev))
      echo "WATCHDOG $(date '+%F %T'): user_total=$((u/1048576))GB campaign=$((c/1048576))GB growth=$((g/1024))MB/5min"
      [ "${u}" -gt "${WARN_KB}" ] && echo "WATCHDOG WARN: user total exceeds 1.5 TiB"
      if [ "${u}" -gt "${KILL_KB}" ] || [ $((u + g)) -gt "${HARD_KB}" ]; then
        echo "WATCHDOG CANCEL: user=$((u/1048576))GB projected=$(((u+g)/1048576))GB -- cancelling ${SLURM_JOB_ID}"
        scancel "${SLURM_JOB_ID}"
      fi
      prev=${u}
      sleep 300
    done ) &
  WATCHDOG_PID=$!
}
