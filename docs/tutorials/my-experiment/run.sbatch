#!/bin/bash
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH --cpus-per-task=30
#SBATCH --exclusive
#SBATCH --job-name=my-experiment
#SBATCH --array=1-10
#SBATCH --mem=1G
#SBATCH -p pml
#SBATCH --output=/Users/artix41/Research/UCL/panqec/docs/tutorials/my-experiment/%j.out

module purge
module load gcc/devtoolset/9
module load anaconda/anaconda3.7
source ~/.bashrc

panqec monitor-usage my-experiment/logs/usage_$SLURM_JOB_ID_$SLURM_ARRAY_TASK_ID.txt &

panqec run-parallel -d /Users/artix41/Research/UCL/panqec/docs/tutorials/my-experiment -n 10 -j $SLURM_ARRAY_TASK_ID -c 30 -t 1000 --delete-existing

date