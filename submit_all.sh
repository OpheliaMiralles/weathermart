#!/bin/bash
START_YEAR=${START_YEAR:-2020}
START_MONTH=${START_MONTH:-05}
END_YEAR=${END_YEAR:-$(date +%Y)}
END_MONTH=${END_MONTH:-$(date +%m)}
MAX_JOBS=${MAX_JOBS:-1}
POLL_SECONDS=${POLL_SECONDS:-60}
CANCEL_ECMWF_ON_EXIT=${CANCEL_ECMWF_ON_EXIT:-1}
SUBMITTED_JOBS=""
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cleanup() {
  status=${1:-$?}
  trap - INT TERM HUP EXIT
  if [ "$status" -ne 0 ] && [ "$CANCEL_ECMWF_ON_EXIT" = "1" ]; then
    echo "Launcher interrupted; cleaning up submitted SGE jobs and ECMWF MARS requests"
    for JOB_ID in $SUBMITTED_JOBS; do
      if [ -n "$JOB_ID" ]; then
        qdel "$JOB_ID" >/dev/null 2>&1 || true
      fi
    done
    python "$SCRIPT_DIR/scripts/cancel_ecmwf_mars_requests.py" queued active submitted || true
  fi
  exit "$status"
}

trap 'cleanup 130' INT
trap 'cleanup 143' TERM
trap 'cleanup 129' HUP
trap 'cleanup $?' EXIT

count_mars_jobs() {
  qstat -u "$USER" | awk '$3 ~ /^mars_/ {count++} END {print count + 0}'
}

wait_for_slot() {
  while [ "$(count_mars_jobs)" -ge "$MAX_JOBS" ]; do
    echo "WAITING: $(count_mars_jobs) mars jobs active, limit ${MAX_JOBS}"
    sleep "$POLL_SECONDS"
  done
}

for YEAR in $(seq ${START_YEAR} ${END_YEAR}); do
  for MONTH in $(seq -w 1 12); do
    if [ "${YEAR}" -eq "${START_YEAR}" ] && [ "${MONTH}" -lt "${START_MONTH}" ]; then
      continue
    fi
    if [ "${YEAR}" -eq "${END_YEAR}" ] && [ "${MONTH}" -gt "${END_MONTH}" ]; then
      continue
    fi
    wait_for_slot
    QSUB_OUTPUT=$(qsub -N mars_${YEAR}_${MONTH} -v YEAR=${YEAR},MONTH=${MONTH} run_month.sh)
    JOB_ID=$(printf "%s\n" "$QSUB_OUTPUT" | awk '{print $3}')
    if [ -n "$JOB_ID" ]; then
      SUBMITTED_JOBS="${SUBMITTED_JOBS} ${JOB_ID}"
    fi
    echo "SUBMITTED MARS: ${YEAR}-${MONTH}"
  done
done

echo "Launcher finished at $(date)"
