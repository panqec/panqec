#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-10:00
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --job-name=sm1
#SBATCH --array=1-6
#SBATCH -p defq
#SBATCH --output=/home/ehuang1/bn3d/slurm/out/statmech/%x_%j.out
set -euxo pipefail

# Variables to change.
data_dir=temp/statmech/test_7
input_dir="$data_dir/inputs"
log_dir="$data_dir/logs"
bash_command="bn3d statmech sample {}"

pwd

module purge
module load python/3.8
source venv/bin/activate

# Print out all environmental variables and date.
printenv
date

# Total number of array jobs.
n_tasks=$SLURM_ARRAY_TASK_COUNT

# Index of this array job.
i_task=$SLURM_ARRAY_TASK_ID

# Monitor CPU usage.
python scripts/monitor.py "$log_dir/usage_${SLURM_JOB_ID}_${i_task}.txt" &

# Function that prints out filtered files so each array job gets a roughly
# equal number of inputs to run.
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
