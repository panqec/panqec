#!/bin/bash
#SBATCH --partition=${partition}
#SBATCH --job-name=${job_name}
#SBATCH --output=${output}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks=${ntasks}
#SBATCH --mem=${mem}
#SBATCH --cpus-per-task=${cpus_per_task}
#SBATCH --time=${time}

set -euxo pipefail

pwd
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
date

panqec run --file ${input_file} --trials ${n_trials}${options}

date
