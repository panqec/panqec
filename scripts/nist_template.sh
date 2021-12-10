#!/bin/bash
#SBATCH -N 1
#SBATCH -t ${TIME}
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --job-name=${NAME}
#SBATCH --array=1-${NARRAY}
#SBATCH --mem=${MEMORY}
#SBATCH -p ${QUEUE}
#SBATCH --output=${DATADIR}/%j.out

module purge
module load gcc/devtoolset/9
module load anaconda/anaconda3.7
module load gnuparallel/20200622

source ~/.bashrc

# Load python and activate the python virtual environment.
conda deactivate
conda activate bn3d

# Variables to change.
trials=${TRIALS}
data_dir=${DATADIR}
n_split=${SPLIT}

# The input directory.
input_dir="$data_dir/inputs"

# The bash command to parallelize.
bash_command="bn3d run -f $input_dir/{1}.json -o $data_dir/{2} -t {3}"

# Print out the current working directory so we know where we are.
pwd

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

# Function that prints out tab-separated values with columns input name,
# results directory name and number of trials to do for that split.
function filter_input {
    counter=0
    
    # Iterate through files in the input directory.
    for filename in $input_dir/*.json; do

        # Just the file name without directory and extension.
        input_name=$(basename -s .json $filename)

        # Each array job gets assigned some input files.
        if [[ $(( counter % n_tasks + 1 )) -eq $i_task ]]; then

            # Name the results directory 'results' if no splitting.
            if [ $n_split -eq 1 ]; then
                results_dir=results
                split_trials=$trials
                echo -e "$input_name\t$results_dir\t$split_trials"

            # Split the results over directories results_1, results_2, etc.
            else
                for i_split in $(seq 1 $n_split); do
                    results_dir="results_$i_split"
                    mkdir -p $data_dir/$results_dir
                    split_trials=$(( (trials - trials % n_split)/n_split ))
                    if [ $i_split -eq $n_split ]; then
                        split_trials=$(( split_trials + trials % n_split ))
                    fi
                    echo -e "$input_name\t$results_dir\t$split_trials"
                done
            fi
        fi
        counter=$(( counter + 1 ))
    done
}

# Write filtered input into GNU parallel to text file in logs.
filter_input  >> $log_dir/filter_${SLURM_JOB_ID}_${i_task}.txt

# Run in parallel.
filter_input | parallel --colsep '\t' --results $log_dir "$bash_command"

# Print out the date when done.
date
