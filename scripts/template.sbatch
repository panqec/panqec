#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=$job_name
#SBATCH --nodes=3
#SBATCH --output=$job_name.out
#SBATCH --time=5:00:00

set -euxo pipefail

pwd
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
date

module load python/3.8
source venv/bin/activate
bn3d run --file scripts/$job_name.json --trials $n_trials

date
