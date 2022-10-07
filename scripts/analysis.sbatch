#!/bin/bash
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --job-name=analysis
#SBATCH --array=1-1
#SBATCH --mem=20G
#SBATCH -p dpart
#SBATCH --qos dpart
#SBATCH --output=/nfshomes/ehuang18/panqec/temp/plots/%j.out

module add Python3/3.8.12

# Load python and activate the python virtual environment.
source venv/bin/activate

plot_dir=temp/plots
log_dir="$plot_dir/logs"
overrides=scripts/overrides.json
paths=temp/final

pwd
df -h
printenv
date

mkdir -p "$plot_dir"
mkdir -p "$log_dir"

# Monitor resource usage.
python scripts/monitor.py "$log_dir/usage_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt" &

# Do the actual analysis.
stdbuf -oL panqec analyze --overrides $overrides --plot_dir $plot_dir $paths \
        2> $log_dir/stderr 1> $log_dir/stdout &

sleep 48h

date
