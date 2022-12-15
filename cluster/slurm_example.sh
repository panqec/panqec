#!/bin/bash

# General parameters
data_dir=/home/ucapacp/panqec/temp/toric-2d-code

# Experiment parameters
code="Toric2DCode" # code class
decoder="MatchingDecoder" # decoder class
sizes="5,7,9"  # lattice sizes
eta="0.5"  # bias parameter
prob="0.1:0.2:0.01" # Physical error rates (format start:end:step)
trials=1000  # number of Monte Carlo runs

# Cluster parameters
slurm_script=$data_dir/run.sbatch
n_cores=8
n_nodes=5
wall_time="00:30:00"
memory="2G"

echo "Generating input"
panqec generate-input -d $data_dir \
       --code_class $code \
       --sizes $sizes \
       --decoder_class $decoder \
       --bias Z \
       --eta $eta \
       --prob $prob

echo "Generating cluster script"
panqec generate-cluster-script \
    cluster/slurm_header.sbatch \
    -o $slurm_script \
    -d $data_dir \
    --cluster slurm \
    -t $trials \
    -n $n_nodes \
    -c $n_cores \
    -w $wall_time \
    -m $memory \
    --delete-existing

echo "Running cluster script"
sbatch $slurm_script