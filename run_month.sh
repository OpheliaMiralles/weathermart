#!/bin/bash
#$ -cwd
#$ -j y
#$ -q bigmem-r8.q
#$ -l h_rt=48:00:00
#$ -l h_data=24G
#$ -o /home/opmir9231/weathermart/logs
date
echo "Starting MARS radiance retrieval for month $YEAR-$MONTH"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cleanup_mars_requests() {
  status=${1:-$?}
  trap - INT TERM HUP EXIT
  if [ "$status" -ne 0 ] && [ "${CANCEL_ECMWF_ON_EXIT:-1}" = "1" ]; then
    echo "MARS month job interrupted; cleaning up ECMWF MARS requests"
    python "$SCRIPT_DIR/scripts/cancel_ecmwf_mars_requests.py" queued active submitted || true
  fi
  exit "$status"
}

trap 'cleanup_mars_requests 130' INT
trap 'cleanup_mars_requests 143' TERM
trap 'cleanup_mars_requests 129' HUP
trap 'cleanup_mars_requests $?' EXIT

source ~/.bashrc
cd /home/opmir9231/weathermart/
source .venv/bin/activate
export MARS_ODB_OUTPUT_DIR=${MARS_ODB_OUTPUT_DIR:-/lustre/storeB/users/opmir9231/tmp/mars_odb_requests}
CA_BUNDLE=$(python -c 'import certifi; print(certifi.where())')
export SSL_CERT_FILE=${SSL_CERT_FILE:-$CA_BUNDLE}
export REQUESTS_CA_BUNDLE=${REQUESTS_CA_BUNDLE:-$CA_BUNDLE}
export CURL_CA_BUNDLE=${CURL_CA_BUNDLE:-$CA_BUNDLE}

python mars_all.py $YEAR $MONTH

date
echo "Finished $YEAR-$MONTH"
