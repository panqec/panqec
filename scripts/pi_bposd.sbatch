#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-10:00
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --job-name=bp1
#SBATCH --array=1-3
#SBATCH -p defq
#SBATCH --output=/home/ehuang1/bn3d/slurm/out/bposd/bias-10-30/%j.out
set -euxo pipefail

# Variables to change.
trials=10000
input_dir=temp/bposd/inputs-bias-10-30
output_dir=temp/bposd/outputs-bias-10-30
log_dir=temp/bposd/logs-bias-10-30
bash_command="bn3d run -f {} -o $output_dir -t $trials"

pwd

module purge
module load python/3.8
source venv/bin/activate

printenv
date
n_tasks=$SLURM_ARRAY_TASK_COUNT
i_task=$SLURM_ARRAY_TASK_ID

python scripts/monitor.py "$log_dir/usage_${SLURM_JOB_ID}_${i_task}.txt" &

# Function that prints out filtered functions.
function filter_files() {
    counter=0
    for filename in $input_dir/*.json; do
        if [[ $(( counter % n_tasks + 1 )) == $i_task ]]; then
            echo $filename
        fi
        counter=$(( counter + 1 ))
    done
}

# Run in parallel.
filter_files | parallel --results $log_dir $bash_command :::

date
