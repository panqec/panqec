paper_dir=temp/paper
mkdir -p "$paper_dir"
qsub_dir=temp/paper/qsub
mkdir -p "$qsub_dir"
mkdir -p temp/paper/share

# ============== Undeformed ==============

name=rotated_toric_splitting_undeformed
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="7,9,11,13,15"
prob="1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,2e-2"
wall_time="24:00:00"
memory="5G"
n_trials=1000000
n_cores=20
n_array=10

panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RotatedToric3DCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "inf" --prob "$prob" --method "splitting"

panqec generate-qsub --data_dir "$paper_dir/$name" --n_array 126 --memory "$memory" \
    --wall_time "$wall_time" --trials $trials --cores 20 "$qsub_dir/$name.qsub"


# ============== Deformed ==============

name=rotated_toric_splitting_deformed
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="7,9,11,13,15"
prob="1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,2e-2"
wall_time="24:00:00"
memory="5G"
n_trials=1000000
n_cores=20
n_array=10

panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RotatedToric3DCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "inf" --prob "$prob" --method "splitting"

panqec generate-qsub --data_dir "$paper_dir/$name" --n_array 126 --memory "$memory" \
    --wall_time "$wall_time" --trials $trials --cores 20 "$qsub_dir/$name.qsub"