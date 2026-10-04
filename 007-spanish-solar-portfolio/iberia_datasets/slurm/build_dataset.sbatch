#!/bin/bash
#-------- Stage 2 on a cluster: build the Anemoi dataset, then check it -------------
# Run after the download array finishes:
#
#   jid=$(sbatch --parsable slurm/download_era5.sbatch)
#   sbatch --dependency=afterok:${jid} slurm/build_dataset.sbatch
#
# For long periods, anemoi-datasets can build in parts (init / load --part i/N /
# finalise) across an array job; for two years over Iberia one task is enough.
#
#SBATCH --job-name=iberia-era5-build
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/build_%j.out
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --qos=YOUR_QOS

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

anemoi-datasets create recipes/iberia-era5.yaml data/iberia-era5.zarr --overwrite
anemoi-datasets inspect data/iberia-era5.zarr
python -m iberia.check data/iberia-era5.zarr
