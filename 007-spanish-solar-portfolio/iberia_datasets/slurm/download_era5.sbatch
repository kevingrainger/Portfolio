#!/bin/bash
#-------- Stage 1 on a cluster: one array task per month --------------------------
# 24 months, 2022-01 to 2023-12. The CDS queues requests, so most of the wall time
# is waiting; tasks need almost no CPU or memory. Limit concurrency (%4) to stay
# polite to the CDS.
#
#   sbatch slurm/download_era5.sbatch
#
#SBATCH --job-name=era5-download
#SBATCH --array=0-23%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=06:00:00
#SBATCH --output=logs/download_%A_%a.out
## Site-specific - set for your cluster:
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --qos=YOUR_QOS

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

python download_era5.py --index "${SLURM_ARRAY_TASK_ID}" 2022-01 2023-12
