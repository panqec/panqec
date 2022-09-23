paper_dir=temp/paper
mkdir -p "$paper_dir"
qsub_dir=temp/paper/qsub
mkdir -p "$qsub_dir"
mkdir -p temp/paper/share

# ============== Undeformed ==============

name=color_3d_undef
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="4,6"
wall_time="12:00:00"
memory="2G"
n_trials=1000
n_cores=10
n_array=20

panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class Color3DCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder_class BeliefPropagationOSDDecoder --bias Z \
    --eta "inf" --prob "0.01:0.3:0.01"

panqec generate-qsub --data_dir "$paper_dir/$name" --n_array $n_array --memory "$memory" \
    --wall_time "$wall_time" --trials $n_trials --cores $n_cores "$qsub_dir/$name.qsub"

# ============== Deformed ==============

# name=rotated_toric_splitting_deformed
# rm -rf $paper_dir/$name/inputs
# rm -rf $paper_dir/$name/logs
# sizes="5,6"
# prob="1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,2e-2"
# wall_time="24:00:00"
# memory="5G"
# n_trials=1000000
# n_cores=10
# n_array=20

# panqec generate-input -i "$paper_dir/$name/inputs" \
#     --code_class RotatedToric3DCode --noise_class DeformedXZZXErrorModel \
#     --ratio coprime \
#     --sizes "$sizes" --decoder_class BeliefPropagationOSDDecoder --bias Z \
#     --eta "inf" --prob "$prob" --method "splitting" --label $name

# panqec generate-qsub --data_dir "$paper_dir/$name" --n_array $n_array --memory "$memory" \
#     --wall_time "$wall_time" --trials $n_trials --cores $n_cores "$qsub_dir/$name.qsub"