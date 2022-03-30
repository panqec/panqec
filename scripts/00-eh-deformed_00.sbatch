#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=deformed_01
#SBATCH --nodes=3
#SBATCH --output=deformed_01.out
#SBATCH --time=20:00:00

set -euxo pipefail

pwd
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
date

module load python/3.8
source venv/bin/activate
panqec run --file scripts/deformed_01.json --trials 1000 --o $TMPDIR

date
