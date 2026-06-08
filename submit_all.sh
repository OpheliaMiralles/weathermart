#!/bin/bash
START_YEAR=${START_YEAR:-2022}
START_MONTH=${START_MONTH:-01}
END_YEAR=${END_YEAR:-$(date +%Y)}
END_MONTH=${END_MONTH:-$(date +%m)}
MAX_JOBS=${MAX_JOBS:-8}
POLL_SECONDS=${POLL_SECONDS:-60}

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
    qsub -N mars_${YEAR}_${MONTH} -v YEAR=${YEAR},MONTH=${MONTH} run_month.sh
    echo "SUBMITTED MARS: ${YEAR}-${MONTH}"
  done
done
