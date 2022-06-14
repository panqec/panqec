#!/bin/bash
#SBATCH --account=${ACCOUNT}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEMORY}
#SBATCH --job-name=${NAME}
#SBATCH --array=1-${NARRAY}
#SBATCH --output=${DATADIR}/${NAME}_%j.out
#SBATCH --mail-user=${EMAIL}
#SBATCH --mail-type=All
set -euxo pipefail

# Variables to change.
trials=${TRIALS}
data_dir=${DATADIR}
n_split=${SPLIT}

if [ "$n_split" = "auto" ]; then
    autosplit=true
else
    autosplit=false
fi

# The input directory.
input_dir="$data_dir/inputs"

# The bash command to parallelize.
bash_command="panqec run -f $input_dir/{1}.json -o $data_dir/{2} -t {3}"

# Print out the current working directory so we know where we are.
pwd

# Load python and activate the python virtual environment.
module purge
module load python/3.8
source venv/bin/activate

# Create the subdirectory for storing logs.
log_dir="$data_dir/logs"
mkdir -p $log_dir

# Print out available disk space.
df -h

# Print out the environmental variables and the time.
printenv
date
n_tasks=$SLURM_ARRAY_TASK_COUNT
i_task=$SLURM_ARRAY_TASK_ID

# Count the number of cores.
n_cores=$( nproc --all )
echo "n_cores=$n_cores"

# Run a CPU and RAM usage logging script in the background.
python scripts/monitor.py "$log_dir/usage_${SLURM_JOB_ID}_${i_task}.txt" &

# Count total number of tasks assigned to this node.
total_tasks_this_node=0
counter=0
for filename in $input_dir/*.json; do
    if [[ $(( counter % n_tasks + 1 )) -eq $i_task ]]; then
        total_tasks_this_node=$(( total_tasks_this_node + 1 ))
    fi
    counter=$(( counter + 1 ))
done

# Function that prints out tab-separated values with columns input name,
# results directory name and number of trials to do for that split.
function filter_input {

    # Counter for total number of tasks.
    counter=0

    # Number of tasks assigned to this node so far.
    tasks_assigned=0
    
    # Iterate through files in the input directory.
    for filename in $input_dir/*.json; do

        # Just the file name without directory and extension.
        input_name=$(basename -s .json $filename)

        # Each array job gets assigned some input files.
        if [[ $(( counter % n_tasks + 1 )) -eq $i_task ]]; then

            # Calculate n_split if set to auto.
            if [ "$autosplit" = "true" ]; then

                # Don't split if there are more tasks than cores.
                if [ "$total_tasks_this_node" -gt "$n_cores" ]; then
                    n_split=1
                else

                    # Number of splits for typical task, rounded down.
                    n_split=$(( n_cores / total_tasks_this_node ))

                    # Remainder is cores left over which can take 1 more split.
                    remainder=$(( n_cores % total_tasks_this_node ))

                    # The first few assigned tasks get an extra split.
                    if [ "$tasks_assigned" -lt "$remainder" ]; then
                        n_split=$(( n_split + 1 ))
                    fi
                fi
            fi

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
            tasks_assigned=$(( tasks_assigned + 1 ))
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
