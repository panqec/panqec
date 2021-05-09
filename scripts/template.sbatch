#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=$job_name
#SBATCH --nodes=$nodes
#SBATCH --output=$out_dir/$job_name.out
#SBATCH --time=$time

set -euxo pipefail

pwd
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
date

module load python/3.8
source venv/bin/activate
bn3d run --file $input_dir/$job_name.json --trials $n_trials

date
