#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-10:00
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --job-name=smt5
#SBATCH --array=1-6
#SBATCH -p defq
#SBATCH --output=/home/ehuang1/bn3d/slurm/out/statmech/test_5_%j.out
set -euxo pipefail

pwd
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
date

module purge
module load python/3.8
source venv/bin/activate
echo "$SLURM_ARRAY_TASK_ID"
bn3d statmech sample --n_jobs 6 temp/statmech/test_5

date
