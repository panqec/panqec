#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=deformed_00
#SBATCH --nodes=3
#SBATCH --output=deformed_00.out
#SBATCH --time=10:00:00

set -euxo pipefail

pwd
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
date

module load python/3.8
source venv/bin/activate
bn3d run --file scripts/deformed_00.json --trials 1000

date
