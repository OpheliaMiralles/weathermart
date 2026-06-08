#!/bin/bash
#$ -cwd
#$ -j y
#$ -q bigmem-r8.q
#$ -l h_rt=48:00:00
#$ -l h_data=24G
#$ -o /home/opmir9231/weathermart/logs
date
echo "Starting EUMETSAT AWS retrieval for month $YEAR-$MONTH"

source ~/.bashrc
cd /home/opmir9231/weathermart/
source .venv/bin/activate

python eumetsat_aws.py $YEAR $MONTH

date
echo "Finished EUMETSAT AWS $YEAR-$MONTH"
