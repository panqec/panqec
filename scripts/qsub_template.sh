#!/bin/bash -l
# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=00:10:00
#$ -l mem=2G
#$ -l tmpfs=1G
#$ -t 1-5
#$ -N toric-2d-code
#$ -pe smp 8
#$ -wd /home/ucapacp/Scratch/output
#Local2Scratch

module purge

source ~/.bashrc

# Load python and activate the python virtual environment.
conda activate panqec

# Variables to change.
data_dir='temp/toric-2d-code'
delete_existing=true
trials=1000
n_cores=8
n_nodes=5
i_node=$SGE_TASK_ID

# Create the subdirectory for storing logs.
log_dir="$data_dir/logs"
mkdir -p $log_dir

# The bash command to parallelize.
run_command="panqec run-parallel -d $experiment_dir -n $n_nodes -j $i_node -c $n_cores -t $n_trials"
if $delete_existing; then
    run_command=run_command" --delete-existing";
fi
monitor_command="panqec monitor-usage "$log_dir/usage_${JOB_ID}_${i_node}.txt" &"

echo "Node: $i_node / $n_nodes"

$monitor_command
$bash_command

date
