#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-20:00
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --job-name=smudzb
#SBATCH --array=1-6
#SBATCH -p defq
#SBATCH --output=/home/ehuang1/panqec/slurm/out/%j.out
set -euxo pipefail

# Variables to change.
trials=10000
data_dir=temp/paper/sweepmatch_undef_zbias
input_dir="$data_dir/inputs"
output_dir="$data_dir/results"
log_dir="$data_dir/logs"
bash_command="panqec run -f {} -o $output_dir -t $trials"

# Print out the current working directory.
pwd

# Load python and activate the python virtual environment.
module purge
module load python/3.8
source venv/bin/activate

# Create the subdirectories
mkdir -p output_dir
mkdir -p log_dir

# Print out the environmental variables and the time.
printenv
date
n_tasks=$SLURM_ARRAY_TASK_COUNT
i_task=$SLURM_ARRAY_TASK_ID

# Run a CPU and RAM usage logging script in the background.
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

# Print out the date when done.
date
