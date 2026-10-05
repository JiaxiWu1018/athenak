#!/bin/bash
# Source inside the allocated simulation job. Only that job can be stopped.
start_storage_watchdog() {
  local root=$1
  (
    set +e
    set +u
    set +o pipefail
    previous_bytes=0
    previous_time=0
    while true; do
      now=$(date +%s)
      usage=$(du -sb /work1/eliasmost/jiaxiwu 2>/dev/null)
      rc=$?
      if [ "$rc" -ne 0 ]; then
        printf 'STORAGE du failed at %s; retrying\n' "$now"
        sleep 300
        continue
      fi
      bytes=${usage%%[[:space:]]*}
      if ! [[ "$bytes" =~ ^[0-9]+$ ]]; then sleep 300; continue; fi
      forecast=$bytes
      if [ "$previous_time" -gt 0 ] && [ "$bytes" -gt "$previous_bytes" ]; then
        # Next ten minutes plus a conservative 8-GiB checkpoint/output burst.
        forecast=$((bytes+(bytes-previous_bytes)*600/(now-previous_time)+8589934592))
      fi
      printf '%s,%s,%s,%s\n' "$now" "$SLURM_JOB_ID" "$bytes" "$forecast" >> "$root/state/storage.csv"
      if [ "$bytes" -ge 1649267441664 ]; then printf 'STORAGE_WARN >=1.5 TiB\n'; fi
      if [ "$bytes" -ge 1869169767219 ] || [ "$forecast" -ge 2089072092774 ]; then
        printf 'STORAGE_STOP whole-root=%s projected=%s; cancel own job %s\n' "$bytes" "$forecast" "$SLURM_JOB_ID"
        printf '%s\n' "STORAGE_STOP job=$SLURM_JOB_ID bytes=$bytes projected=$forecast" > "$root/state/storage_stop.txt"
        scancel "$SLURM_JOB_ID"
        exit
      fi
      previous_bytes=$bytes
      previous_time=$now
      sleep 300
    done
  ) &
  STORAGE_PID=$!
}
