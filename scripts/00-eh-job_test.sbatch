#!/bin/bash
#SBATCH --partition=debugq
#SBATCH --job-name=job_test
#SBATCH --nodes=1
#SBATCH --output=job_test.out
#SBATCH --time=0:10:00

set -euxo pipefail

pwd
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
date

source venv/bin/activate
module load python/3.8


date
