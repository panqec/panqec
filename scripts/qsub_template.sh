#!/bin/bash -l
# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=${TIME}
#$ -l mem=${MEMORY}
#$ -l tmpfs=1G
#$ -N RhombicJob
#$ -wd /home/ucapacp/Scratch/output
#Local2Scratch

module purge
module load parallel/20181122

source ~/.bashrc

# Load python and activate the python virtual environment.
conda deactivate
conda activate bn3d

# Variables to change.
trials=${TRIALS}
data_dir=${DATADIR}
n_split=${SPLIT}
script_dir='/home/ucapacp/bn3d/scripts'

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

SGE_TASK_COUNT=$(($SGE_TASK_LAST-$SGE_TASK_FIRST+1))
n_tasks=$SGE_TASK_COUNT
i_task=$SGE_TASK_ID

# Run a CPU and RAM usage logging script in the background.
python $script_dir/monitor.py "$log_dir/usage_${JOB_ID}_${i_task}.txt" &

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
filter_input  >> $log_dir/filter_${JOB_ID}_${i_task}.txt

# Run in parallel.
filter_input | parallel --colsep '\t' --results $log_dir "$bash_command"

# Print out the date when done.
date
