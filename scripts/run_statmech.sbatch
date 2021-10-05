#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-8:00
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --job-name=stat_mech-array
#SBATCH --array=1-3
#SBATCH -p defq
#SBATCH --output=/home/ehuang1/bn3d/slurm/out/statmech/%j.out
set -euxo pipefail

pwd
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
date

module purge
module load python/3.8
source venv/bin/activate
python3 scripts/run_statmech.py "temp/statmech/test_$SLURM_ARRAY_TASK_ID"

date
