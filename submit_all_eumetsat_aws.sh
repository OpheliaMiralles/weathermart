#!/bin/bash
YEAR=${YEAR:-2026}
MAX_JOBS=2

for MONTH in $(seq -w 1 5); do
  while [ "$(qstat -u "$USER" | grep -c "aws_${YEAR}_")" -ge "$MAX_JOBS" ]; do
    sleep 60
  done

  qsub -N aws_${YEAR}_${MONTH} -v YEAR=${YEAR},MONTH=${MONTH} run_month_eumetsat_aws.sh
  echo "SUBMITTED EUMETSAT AWS: ${YEAR}-${MONTH}"
done