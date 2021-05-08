#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes=3
#SBATCH --output=$JOB_NAME.out
#SBATCH --time=5:00:00

set -euxo pipefail

pwd
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
date

module load python/3.8
source venv/bin/activate
bn3d run --file scripts/$JOB_NAME.json --trials $N_TRIALS

date
