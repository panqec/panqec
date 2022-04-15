paper_dir=temp/paper
mkdir -p "$paper_dir"
sbatch_dir=temp/paper/sbatch
mkdir -p "$sbatch_dir"
mkdir -p temp/paper/share

wall_time="24:00:00"
memory="64GB"

# # Rough runs using BPOSD decoder on toric code
# for repeat in $(seq 1 6); do
#     name=unrot_bposd_xzzx_zbias_$repeat
#     rm -rf $paper_dir/$name/inputs
#     rm -rf $paper_dir/$name/logs
#     bn3d generate-input -i "$paper_dir/$name/inputs" \
#         --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
#         --sizes "9,13,17,21" --decoder BeliefPropagationOSDDecoder --bias Z \
#         --eta "30,100" --prob "0:0.5:0.02"
#     bn3d cc-sbatch --data_dir "$paper_dir/$name" --n_array 50 --memory $memory \
#         --wall_time "$wall_time" --trials 1667 --split 1 $sbatch_dir/$name.sbatch
# 
#     name=unrot_bposd_undef_zbias_$repeat
#     rm -rf $paper_dir/$name/inputs
#     rm -rf $paper_dir/$name/logs
#     bn3d generate-input -i "$paper_dir/$name/inputs" \
#         --lattice kitaev --boundary toric --deformation none --ratio equal \
#         --sizes "9,13,17,21" --decoder BeliefPropagationOSDDecoder --bias Z \
#         --eta "30,100" --prob "0:0.5:0.02"
#     bn3d cc-sbatch --data_dir "$paper_dir/$name" --n_array 50 --memory $memory \
#         --wall_time "$wall_time" --trials 1667 --split 1 $sbatch_dir/$name.sbatch
# done

# # Subthreshold scaling coprime 4k+2 runs for scaling with distance.
# for repeat in $(seq 1 10); do
#     name=sts_coprime_scaling_eta300_$repeat
#     rm -rf $paper_dir/$name/inputs
#     rm -rf $paper_dir/$name/logs
#     sizes="6,10,14,18"
#     bn3d generate-input -i "$paper_dir/$name/inputs" \
#         --code_class LayeredRotatedToricCode --noise_class DeformedXZZXErrorModel \
#         --ratio coprime \
#         --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#         --eta "inf" --prob "0.08:0.12:0.01"
#     bn3d cc-sbatch --data_dir "$paper_dir/$name" --n_array 5 --memory $memory \
#         --wall_time "$wall_time" --trials 50000 --split 32 $sbatch_dir/$name.sbatch
# done
# 
# # Threshold runs for coprime 4k+2.
# name=thr_coprime_scaling_eta300
# rm -rf $paper_dir/$name/inputs
# rm -rf $paper_dir/$name/logs
# sizes="6,10,14,18"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class LayeredRotatedToricCode --noise_class DeformedXZZXErrorModel \
#     --ratio coprime \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "300" --prob "0.20:0.40:0.02"
# bn3d cc-sbatch --data_dir "$paper_dir/$name" --n_array 12 --memory $memory \
#     --wall_time "$wall_time" --trials 10000 --split 32 $sbatch_dir/$name.sbatch



# Extra rhombic
common_name=det_rhombic_bposd_xzzx_xbias_extra
name=${common_name}_temp
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="10,14,18,20"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "1000" --prob "0.50"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "inf" --prob "0.30:0.50:0.02"

for repeat in $(seq 1 40); do
    name=${common_name}_${repeat}
    mkdir -p $paper_dir/$name
    rm -rf $paper_dir/$name/inputs
    rm -rf $paper_dir/$name/logs
    cp -R $paper_dir/${common_name}_temp/inputs $paper_dir/$name/inputs
    bn3d cc-sbatch --data_dir "$paper_dir/$name" --n_array 13 --memory "$memory" \
        --wall_time "$wall_time" --trials 250 --split 4 $sbatch_dir/$name.sbatch
done

rm -rf $paper_dir/${common_name}_temp
