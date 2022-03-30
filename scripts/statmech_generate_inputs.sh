statmech_dir=temp/statmech
mkdir -p "$statmech_dir"
sbatch_dir=temp/statmech/sbatch
mkdir -p "$sbatch_dir"
targets_dir=temp/statmech/targets
mkdir -p "$targets_dir"

ratio=equal
wall_time="0-00:59"
queue=debugq

# Example run.
name=test_9
data_dir=$statmech_dir/$name
panqec statmech generate $data_dir --targets $targets_dir/$name.json
panqec statmech pi-sbatch --data_dir $data_dir \
    --n_array 6 --queue $queue --wall_time $wall_time \
    $sbatch_dir/$name.sbatch
