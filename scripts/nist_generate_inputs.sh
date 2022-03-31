paper_dir=temp/paper
mkdir -p "$paper_dir"
sbatch_dir=temp/paper/sbatch
mkdir -p "$sbatch_dir"
mkdir -p temp/paper/share

# # Sweepmatch undeformed
# name=unrot_sweepmatch_undef_zbias
# rm -rf $paper_dir/$name/inputs
# rm -rf $paper_dir/$name/logs
# sizes="9,11,13,17,21"
# wall_time="5-00:00"
# memory="93GB"
# queue=pml
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class ToricCode3D --noise_class PauliErrorModel \
#     --ratio equal \
#     --sizes "$sizes" --decoder SweepMatchDecoder --bias Z \
#     --eta "0.5,1" --prob "0:0.1:0.01"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class ToricCode3D --noise_class PauliErrorModel \
#     --ratio equal \
#     --sizes "$sizes" --decoder SweepMatchDecoder --bias Z \
#     --eta "3" --prob "0.08:0.18:0.01"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class ToricCode3D --noise_class PauliErrorModel \
#     --ratio equal \
#     --sizes "$sizes" --decoder SweepMatchDecoder --bias Z \
#     --eta "10,30,100,1000,inf" --prob "0.1:0.2:0.01"
# bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 93 --partition $queue \
#     --memory "$memory" \
#     --wall_time "$wall_time" --trials 10000 --split 40 $sbatch_dir/$name.sbatch
# 
# # Sweepmatch XZZX
# name=unrot_sweepmatch_xzzx_zbias
# rm -rf $paper_dir/$name/inputs
# rm -rf $paper_dir/$name/logs
# sizes="9,11,13,17,21"
# wall_time="5-00:00"
# memory="93GB"
# queue=pml
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class ToricCode3D --noise_class DeformedXZZXErrorModel \
#     --ratio equal \
#     --sizes "$sizes" --decoder DeformedSweepMatchDecoder --bias Z \
#     --eta "0.5,1" --prob "0:0.1:0.01"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class ToricCode3D --noise_class DeformedXZZXErrorModel \
#     --ratio equal \
#     --sizes "$sizes" --decoder DeformedSweepMatchDecoder --bias Z \
#     --eta "3" --prob "0.03:0.13:0.01"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class ToricCode3D --noise_class DeformedXZZXErrorModel  \
#     --ratio equal \
#     --sizes "$sizes" --decoder DeformedSweepMatchDecoder --bias Z \
#     --eta "10" --prob "0.09:0.19:0.01"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class ToricCode3D --noise_class DeformedXZZXErrorModel \
#     --ratio equal \
#     --sizes "$sizes" --decoder DeformedSweepMatchDecoder --bias Z \
#     --eta "30,100,1000,inf" --prob "0.16:0.26:0.01"
# bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 94 --partition $queue \
#     --memory "$memory" \
#     --wall_time "$wall_time" --trials 10000 --split 40 $sbatch_dir/$name.sbatch


# # Subthreshold scaling coprime 4k+2 runs for scaling with distance.
# for repeat in $(seq 1 10); do
#     name=sts_coprime_scaling_$repeat
#     rm -rf $paper_dir/$name/inputs
#     rm -rf $paper_dir/$name/logs
#     sizes="6,10,14,18"
#     wall_time="3-00:00"
#     memory="93GB"
#     queue=pml
#     bn3d generate-input -i "$paper_dir/$name/inputs" \
#         --code_class LayeredRotatedToricCode --noise_class DeformedXZZXErrorModel \
#         --ratio coprime \
#         --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#         --eta "300" --prob "0.19:0.23:0.01"
#     bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 6 --partition $queue \
#         --memory "$memory" \
#         --wall_time "$wall_time" --trials 50000 --split 40 $sbatch_dir/$name.sbatch
# done
# 
# # Threshold runs for coprime 4k+2.
# name=thr_coprime_scaling
# rm -rf $paper_dir/$name/inputs
# rm -rf $paper_dir/$name/logs
# sizes="6,10,14,18"
# wall_time="5-00:00"
# memory="93GB"
# queue=pml
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class LayeredRotatedToricCode --noise_class DeformedXZZXErrorModel \
#     --ratio coprime \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "30,100,inf" --prob "0:0.55:0.05"
# bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 39 --partition $queue \
#     --memory "$memory" \
#     --wall_time "$wall_time" --trials 10000 --split 40 $sbatch_dir/$name.sbatch

# Final runs.
# name=det_unrot_bposd_xzzx_zbias
# rm -rf $paper_dir/$name/inputs
# rm -rf $paper_dir/$name/logs
# sizes="21"
# wall_time="5-00:00"
# memory="93GB"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "0.5" --prob "0.055:0.075:0.002"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "3" --prob "0.072:0.092:0.002"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "10" --prob "0.128:0.148:0.002"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "30" --prob "0.154:0.194:0.004"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "100" --prob "0.170:0.210:0.004"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "1000" --prob "0.18:0.28:0.01"
# bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 12 --queue $queue \
#     --wall_time "$wall_time" --trials 10000 --split 2 $sbatch_dir/$name.sbatch
# 
# name=det_unrot_bposd_undef_zbias
# rm -rf $paper_dir/$name/inputs
# rm -rf $paper_dir/$name/logs
# sizes="21"
# wall_time="5-00:00"
# memory="93GB"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --lattice rotated --boundary planar --deformation none --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "0.5" --prob "0.055:0.075:0.002"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --lattice rotated --boundary planar --deformation none --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "3" --prob "0.130:0.150:0.002"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --lattice rotated --boundary planar --deformation none --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "10" --prob "0.226:0.246:0.002"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --lattice rotated --boundary planar --deformation none --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "30" --prob "0.213:0.233:0.002"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --lattice rotated --boundary planar --deformation none --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "100" --prob "0.210:0.230:0.002"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --lattice rotated --boundary planar --deformation none --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "1000" --prob "0.210:0.230:0.002"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --lattice rotated --boundary planar --deformation none --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "inf" --prob "0.209:0.229:0.002"
# bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 14 --queue $queue \

# name=det_rhombic_bposd_undef_xbias
# rm -rf $paper_dir/$name/inputs
# rm -rf $paper_dir/$name/logs
# sizes="10,14,18,22"
# wall_time="14-00:00"
# memory="90GB"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class RhombicCode --noise_class PauliErrorModel \
#     --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
#     --eta "0.5" --prob "0.010:0.020:0.001"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class RhombicCode --noise_class PauliErrorModel \
#     --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
#     --eta "3" --prob "0.026:0.046:0.002"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class RhombicCode --noise_class PauliErrorModel \
#     --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
#     --eta "10" --prob "0.070:0.130:0.005"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class RhombicCode --noise_class PauliErrorModel \
#     --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
#     --eta "20" --prob "0.15:0.22:0.006"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class RhombicCode --noise_class PauliErrorModel \
#     --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
#     --eta "30" --prob "0.25:0.32:0.006"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class RhombicCode --noise_class PauliErrorModel \
#     --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
#     --eta "100,inf" --prob "0.25:0.30:0.005"
# bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 42 --memory "$memory" \
#     --wall_time "$wall_time" --trials 10000 --split 20 $sbatch_dir/$name.sbatch
# 
# name=det_rhombic_bposd_xzzx_xbias
# rm -rf $paper_dir/$name/inputs
# rm -rf $paper_dir/$name/logs
# sizes="10,14,18,22"
# wall_time="14-00:00"
# memory="90GB"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
#     --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
#     --eta "0.5" --prob "0.010:0.020:0.001"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
#     --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
#     --eta "3" --prob "0.025:0.035:0.001"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
#     --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
#     --eta "10" --prob "0.05:0.07:0.002"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
#     --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
#     --eta "30" --prob "0.07:0.11:0.004"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
#     --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
#     --eta "100" --prob "0.09:0.18:0.01"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
#     --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
#     --eta "1000" --prob "0.10:0.50:0.02"
# bn3d generate-input -i "$paper_dir/$name/inputs" \
#     --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
#     --ratio equal \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
#     --eta "inf" --prob "0.30:0.50:0.02"
# bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 45 --memory "$memory" \
#     --wall_time "$wall_time" --trials 10000 --split 8 $sbatch_dir/$name.sbatch


# Extra rhombic
for repeat in $(seq 1 5); do
    name=det_rhombic_bposd_xzzx_xbias_extra_$repeat
    rm -rf $paper_dir/$name/inputs
    rm -rf $paper_dir/$name/logs
    sizes="10,14,18,22"
    wall_time="7-00:00"
    memory="90GB"
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
    bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 13 --memory "$memory" \
        --wall_time "$wall_time" --trials 2000 --split 4 $sbatch_dir/$name.sbatch
done
