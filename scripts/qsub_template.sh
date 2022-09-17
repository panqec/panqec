#!/bin/bash -l
# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=${TIME}
#$ -l mem=${MEMORY}
#$ -l tmpfs=10G
#$ -t 1-${NARRAY}
#$ -N ${NAME}
#$ -pe smp ${CORES}
#$ -wd /home/ucapacp/Scratch/output
#Local2Scratch

module purge
module load parallel/20181122

source ~/.bashrc

# Load python and activate the python virtual environment.
conda deactivate
conda activate panqec

# Variables to change.
trials=${TRIALS}
data_dir=${DATADIR}
n_cores=${CORES}
script_dir='/home/ucapacp/panqec/scripts'

# The input directory.
input_dir="$data_dir/inputs"

# The bash command to parallelize.
bash_command="panqec run -f $input_dir/{1}.json -o $data_dir/{2} -t {3}"

# Print out the current working directory so we know where we are.
pwd

# Create the subdirectory for storing logs.
log_dir="$data_dir/logs"
mkdir -p $log_dir

# Print out the environmental variables and the time.
# printenv
# date

n_nodes=${NARRAY}
i_node=$SGE_TASK_ID
list_inputs=( $(ls $input_dir/*.json) )
n_inputs=${#list_inputs[@]}
n_tasks=$(( n_nodes * n_cores ))

echo "Node: $i_node / $n_nodes"

# Run a CPU and RAM usage logging script in the background.
python $script_dir/monitor.py "$log_dir/usage_${JOB_ID}_${i_node}.txt" &

# Function that prints out tab-separated values with columns input name,
# results directory name and number of trials to do for that split.
function filter_input {
    for i_core in $( seq 0 $(( n_cores-1 )) ); do
        i_task=$((n_cores * i_node + i_core))

        n_tasks_per_input=$(( n_tasks / n_inputs ))

        i_input=$(( i_task / n_tasks_per_input ))
        i_input=$(( i_input < n_inputs ? i_input : n_inputs-1 ))

        if [ $i_input -eq $(( n_inputs-1 )) ]; then
            n_tasks_per_input=$(( n_tasks_per_input + n_tasks % n_inputs ))
        fi

        i_task_in_input=$(( i_task % n_tasks_per_input ))

        if [ $i_input -eq $(( n_inputs-1 )) ]; then
            i_task_in_input=$(( i_task - n_tasks / n_inputs * (n_inputs - 1) ))
        fi

        n_runs=$(( trials / n_tasks_per_input ))

        if [ $i_task_in_input -eq $(( n_tasks_per_input-1 )) ]; then
            n_runs=$(( n_runs +     trials % n_runs ))
        fi

        filename=${list_inputs[i_input]}
        input_name=$(basename -s .json $filename)

        # Split the results over directories results_1, results_2, etc.
        results_dir="results_$i_task"
        mkdir -p $data_dir/$results_dir

        echo -e "$input_name\t$results_dir\t$n_runs"
    done
}

# Write filtered input into GNU parallel to text file in logs.
filter_input  >> $log_dir/filter_${JOB_ID}_${i_node}.txt

# Run in parallel.
filter_input | parallel --colsep '\t' --results $log_dir "$bash_command"

# Print out the date when done.
date
