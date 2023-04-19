#!/bin/bash
#SBATCH -N 1
#SBATCH -t ${TIME}
#SBATCH --cpus-per-task=${N_CORES}
#SBATCH --exclusive
#SBATCH --job-name=${NAME}
#SBATCH --array=1-${N_NODES}
#SBATCH --mem=${MEMORY}
#SBATCH --partition=${QUEUE}
#SBATCH --qos=${QOS}
#SBATCH --output=${DATA_DIR}/%j.out

module purge
module load gcc/devtoolset/9
module load anaconda/anaconda3.7
source ~/.bashrc
