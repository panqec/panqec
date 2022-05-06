#!/bin/bash
#SBATCH -N 1
#SBATCH -t ${TIME}
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --job-name=${NAME}
#SBATCH --array=1-${NARRAY}
#SBATCH -p ${QUEUE}
#SBATCH --output=/home/ehuang1/panqec/slurm/out/${NAME}_%j.out
set -euxo pipefail

# Variables to change.
data_dir=${DATADIR}
n_split=${SPLIT}

# The input directory.
input_dir="$data_dir/inputs"

# The bash command to parallelize.
bash_command="panqec statmech sample $data_dir/inputs/input_{1}.gz"

# Print out the current working directory so we know where we are.
pwd

# Load python and activate the python virtual environment.
module purge
module load python/3.8
source venv/bin/activate

# Create the subdirectory for storing logs.
log_dir="$data_dir/logs"
mkdir -p $log_dir

# Print out the environmental variables and the time.
printenv
date
n_tasks=$SLURM_ARRAY_TASK_COUNT
i_task=$SLURM_ARRAY_TASK_ID

# Run a CPU and RAM usage logging script in the background.
python scripts/monitor.py "$log_dir/usage_${SLURM_JOB_ID}_${i_task}.txt" &

hashes_txt="$log_dir/filter_${SLURM_JOB_ID}_${i_task}.txt"

# Print out filtered hashes to run on each line.
panqec statmech assign-inputs $data_dir $i_task $n_tasks >> $hashes_txt

# Run in parallel.
cat $hashes_txt | parallel --colsep '\t' --results $log_dir "$bash_command"

# Print out the date when done.
date
