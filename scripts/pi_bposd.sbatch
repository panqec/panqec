#!/bin/bash
#SBATCH -N 1
#SBATCH -t 0-00:10
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --job-name=bposd
#SBATCH --array=1-6
#SBATCH -p debugq
#SBATCH --output=/home/ehuang1/bn3d/slurm/out/bposd/bias-10-30/%j.out
set -euxo pipefail

# Variables to change.
n_samples=50
input_dir=temp/bposd/inputs-bias-10-30
output_dir=temp/bposd/outputs-bias-10-30
log_dir=temp/bposd/logs-bias-10-30

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

filter_files | parallel --results $log_dir bn3d run -f {} -o $output_dir -t $n_samples :::

# for filename in "${filtered_files[@]}"; do
#     log_file=$(basename -- $filename)
#     log_file="$log_dir/${log_file%.*}.txt"
#     echo "$filename" > $log_file
#     # bn3d run -f $filename -o $output_dir -t $n_samples &> "$log_file" &
# done

# printf '%s\n' "${filtered_files[@]}"

date
