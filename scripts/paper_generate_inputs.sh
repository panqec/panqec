paper_dir=temp/paper
mkdir -p "$paper_dir"
sbatch_dir=temp/paper/sbatch
mkdir -p "$sbatch_dir"
mkdir -p temp/paper/share

ratio=equal
wall_time="0-23:59"
queue=defq

# # Subthreshold scaling coprime 4k+2 runs for scaling with distance.
# for repeat in $(seq 1 10); do
#     name=sts_coprime_scaling_$repeat
#     rm -rf $paper_dir/$name/inputs
#     rm -rf $paper_dir/$name/logs
#     sizes="6,10,14,18"
#     panqec generate-input -i "$paper_dir/$name/inputs" \
#         --code_class LayeredRotatedToricCode --noise_class DeformedXZZXErrorModel \
#         --ratio coprime \
#         --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#         --eta "inf" --prob "0.45:0.49:0.01"
#     panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 6 --queue $queue \
#         --wall_time "$wall_time" --trials 50000 --split 40 $sbatch_dir/$name.sbatch
# done

# Final runs.
name=det_unrot_bposd_xzzx_zbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="9,11,13,17"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "0.5" --prob "0.055:0.075:0.002"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "3" --prob "0.072:0.092:0.002"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "10" --prob "0.128:0.148:0.002"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "30" --prob "0.154:0.194:0.004"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "100" --prob "0.170:0.210:0.004"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "1000" --prob "0.18:0.28:0.01"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 12 --queue $queue \
    --wall_time "$wall_time" --trials 10000 --split 2 $sbatch_dir/$name.sbatch

# Threshold runs for coprime 4k+2.
name=thr_coprime_scaling
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="6,10,14,18"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class LayeredRotatedToricCode --noise_class DeformedXZZXErrorModel \
    --ratio coprime \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "1000" --prob "0.20:0.55:0.05"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 39 --queue $queue \
    --wall_time "$wall_time" --trials 10000 --split 40 $sbatch_dir/$name.sbatch

# # Final runs.
# name=det_unrot_bposd_xzzx_zbias
# rm -rf $paper_dir/$name/inputs
# rm -rf $paper_dir/$name/logs
# sizes="9,11,13,17,21"
# panqec generate-input -i "$paper_dir/$name/inputs" \
#     --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "0.5" --prob "0.050:0.070:0.002"
# panqec generate-input -i "$paper_dir/$name/inputs" \
#     --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "3" --prob "0.070:0.090:0.002"
# panqec generate-input -i "$paper_dir/$name/inputs" \
#     --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "10" --prob "0.100:0.140:0.004"
# panqec generate-input -i "$paper_dir/$name/inputs" \
#     --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "30" --prob "0.154:0.194:0.004"
# panqec generate-input -i "$paper_dir/$name/inputs" \
#     --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "100" --prob "0.190:0.230:0.004"
# panqec generate-input -i "$paper_dir/$name/inputs" \
#     --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "1000" --prob "0.18:0.28:0.01"
# panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 6 --queue $queue \
#     --wall_time "$wall_time" --trials 10000 --split 7 $sbatch_dir/$name.sbatch
# 
# name=det_unrot_bposd_undef_zbias
# rm -rf $paper_dir/$name/inputs
# rm -rf $paper_dir/$name/logs
# panqec generate-input -i "$paper_dir/$name/inputs" \
#     --lattice kitaev --boundary toric --deformation none --ratio equal  \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "0.5" --prob "0.050:0.070:0.002"
# panqec generate-input -i "$paper_dir/$name/inputs" \
#     --lattice kitaev --boundary toric --deformation none --ratio equal  \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "3" --prob "0.118:0.130:0.002"
# panqec generate-input -i "$paper_dir/$name/inputs" \
#     --lattice kitaev --boundary toric --deformation none --ratio equal  \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "10" --prob "0.214:0.234:0.002"
# panqec generate-input -i "$paper_dir/$name/inputs" \
#     --lattice kitaev --boundary toric --deformation none --ratio equal  \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "30" --prob "0.208:0.228:0.002"
# panqec generate-input -i "$paper_dir/$name/inputs" \
#     --lattice kitaev --boundary toric --deformation none --ratio equal  \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "100" --prob "0.206:0.226:0.002"
# panqec generate-input -i "$paper_dir/$name/inputs" \
#     --lattice kitaev --boundary toric --deformation none --ratio equal  \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "1000" --prob "0.204:0.224:0.002"
# panqec generate-input -i "$paper_dir/$name/inputs" \
#     --lattice kitaev --boundary toric --deformation none --ratio equal  \
#     --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
#     --eta "inf" --prob "0.204:0.224:0.002"
# panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 7 --queue $queue \
#     --wall_time "$wall_time" --trials 10000 --split 7 $sbatch_dir/$name.sbatch
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation none --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "0.5" --prob "0.055:0.075:0.002"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation none --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "3" --prob "0.130:0.150:0.002"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation none --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "10" --prob "0.226:0.246:0.002"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation none --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "30" --prob "0.213:0.233:0.002"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation none --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "100" --prob "0.210:0.230:0.002"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation none --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "1000" --prob "0.210:0.230:0.002"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation none --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "inf" --prob "0.209:0.229:0.002"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 14 --queue $queue \
    --wall_time "$wall_time" --trials 10000 --split 2 $sbatch_dir/$name.sbatch

: '
# Regime where finite-size scaling starts to break down
name=rot_bposd_xzzx_zbias
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation xzzx --ratio "$ratio" \
    --sizes "2,4,6,8,10" --decoder BeliefPropagationOSDDecoder  --bias Z \
    --eta "60,70,80" --prob "0.30:0.40:0.01"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 10 --queue $queue \
    --wall_time "$wall_time" --trials 10000 --split 10 $sbatch_dir/$name.sbatch

# Rough runs for new deformed rhombic code
name=det_rhombic_bposd_undef_zbias
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio "$ratio" \
    --sizes "2,4,6,8" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "0.5,1,10,30,100,inf" --prob "0.005:0.025:0.001"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 6 --queue $queue \
    --wall_time "$wall_time" --trials 10000 --split 10 $sbatch_dir/$name.sbatch

name=det_rhombic_bposd_xzzx_zbias
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio "$ratio" \
    --sizes "2,4,6,8" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "0.5,1,10,30,100,inf" --prob "0.005:0.025:0.001"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 6 --queue $queue \
    --wall_time "$wall_time" --trials 10000 --split 10 $sbatch_dir/$name.sbatch

# Detailed rhombic code runs. These are final.
name=det_rhombic_bposd_undef_xbias
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio "$ratio" \
    --sizes "6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "0.5,1" --prob "0.006:0.025:0.002"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio "$ratio" \
    --sizes "6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "3" --prob "0.02:0.06:0.005"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio "$ratio" \
    --sizes "6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "10" --prob "0.08:0.12:0.005"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio "$ratio" \
    --sizes "6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "15,20,25" --prob "0.10:0.32:0.01"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio "$ratio" \
    --sizes "6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "30,100,inf" --prob "0.26:0.32:0.005"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 6 --queue $queue \
    --wall_time "$wall_time" --trials 10000 --split 5 $sbatch_dir/$name.sbatch

name=det_rhombic_bposd_xzzx_xbias
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio "$ratio" \
    --sizes "6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "0.5,1" --prob "0.01:0.025:0.002"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio "$ratio" \
    --sizes "6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "3" --prob "0.014:0.04:0.002"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio "$ratio" \
    --sizes "6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "10" --prob "0.04:0.08:0.005"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio "$ratio" \
    --sizes "6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "30" --prob "0.10:0.14:0.005"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio "$ratio" \
    --sizes "6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "100" --prob "0.16:0.22:0.005"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio "$ratio" \
    --sizes "6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "inf" --prob "0.38:0.42:0.005"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 6 --queue $queue \
    --wall_time "$wall_time" --trials 10000  --split 5 $sbatch_dir/$name.sbatch

# Subthreshold scaling.
name=lay_coprime_xzzx_zbias
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RotatedToric3DCode --noise_class DeformedXZZXErrorModel \
    --ratio coprime \
    --sizes "4,6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "0.5,10,100,inf" --prob "0:0.5:0.02"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 6 --queue $queue \
    --wall_time "$wall_time" --trials 10000 --split 10 $sbatch_dir/$name.sbatch

name=lay_coprime_undef_zbias
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RotatedToric3DCode --noise_class PauliErrorModel \
    --ratio coprime \
    --sizes "4,6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "0.5,10,100,inf" --prob "0:0.5:0.02"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 6 --queue $queue \
    --wall_time "$wall_time" --trials 10000 --split 10 $sbatch_dir/$name.sbatch

name=lay_equal_xzzx_zbias
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RotatedToric3DCode --noise_class DeformedXZZXErrorModel \
    --ratio equal \
    --sizes "4,6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "0.5,10,100,inf" --prob "0:0.5:0.02"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 6 --queue $queue \
    --wall_time "$wall_time" --trials 10000 --split 10 $sbatch_dir/$name.sbatch

name=lay_equal_undef_zbias
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RotatedToric3DCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "4,6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "0.5,10,100,inf" --prob "0:0.5:0.02"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 6 --queue $queue \
    --wall_time "$wall_time" --trials 10000 --split 10 $sbatch_dir/$name.sbatch


name=sts_lay_coprime_xzzx_zbias_300_1
rm -rf "$paper_dir/$name/inputs"
rm -rf "$paper_dir/$name/logs"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RotatedToric3DCode --noise_class DeformedXZZXErrorModel \
    --ratio coprime \
    --sizes "6,7,9,10,11,13,15" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "300" --prob "0.19:0.23:0.01"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 5 --queue $queue \
    --wall_time "$wall_time" --trials 166667 --split 80 $sbatch_dir/$name.sbatch

name=sts_lay_coprime_xzzx_zbias_300_2
rm -rf "$paper_dir/$name/inputs"
rm -rf "$paper_dir/$name/logs"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RotatedToric3DCode --noise_class DeformedXZZXErrorModel \
    --ratio coprime \
    --sizes "6,7,9,10,11,13,15" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "300" --prob "0.19:0.23:0.01"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 5 --queue $queue \
    --wall_time "$wall_time" --trials 166667 --split 80 $sbatch_dir/$name.sbatch

name=sts_lay_coprime_xzzx_zbias_300_3
rm -rf "$paper_dir/$name/inputs"
rm -rf "$paper_dir/$name/logs"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RotatedToric3DCode --noise_class DeformedXZZXErrorModel \
    --ratio coprime \
    --sizes "6,7,9,10,11,13,15" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "300" --prob "0.19:0.23:0.01"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 5 --queue $queue \
    --wall_time "$wall_time" --trials 166667 --split 80 $sbatch_dir/$name.sbatch


name=sts_lay_coprime_undef_zbias
rm -rf "$paper_dir/$name/inputs"
rm -rf "$paper_dir/$name/logs"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RotatedToric3DCode --noise_class PauliErrorModel \
    --ratio coprime \
    --sizes "7,9,11,13,15" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "100" --prob "0.19:0.23:0.01"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 4 --queue $queue \
    --wall_time "$wall_time" --trials 500000 --split 80 $sbatch_dir/$name.sbatch

name=sts_lay_equal_xzzx_zbias
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RotatedToric3DCode --noise_class DeformedXZZXErrorModel \
    --ratio equal \
    --sizes "2,4,6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "100" --prob "0.1"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 6 --queue $queue \
    --wall_time "$wall_time" --trials 100000 --split 90 $sbatch_dir/$name.sbatch

name=sts_lay_equal_undef_zbias
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RotatedToric3DCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "2,4,6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "100" --prob "0.1"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 6 --queue $queue \
    --wall_time "$wall_time" --trials 100000 --split 90 $sbatch_dir/$name.sbatch

# Main runs Z bias
name=det_rot_bposd_undef_zbias 
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation none --ratio "$ratio" \
    --sizes "6,9,12,15" --decoder BeliefPropagationOSDDecoder   --bias Z \
    --eta "0.5" --prob "0.04:0.09:0.005"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation none --ratio "$ratio" \
    --sizes "6,9,12,15" --decoder BeliefPropagationOSDDecoder   --bias Z \
    --eta "1" --prob "0.05:0.10:0.005"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation none --ratio "$ratio" \
    --sizes "6,9,12,15" --decoder BeliefPropagationOSDDecoder   --bias Z \
    --eta "3" --prob "0.115:0.165:0.005"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation none --ratio "$ratio" \
    --sizes "6,9,12,15" --decoder BeliefPropagationOSDDecoder   --bias Z \
    --eta "10,30,100,inf" --prob "0.19:0.26:0.005"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 10 --queue $queue \
    --wall_time "$wall_time" --trials 10000 --split 1 $sbatch_dir/$name.sbatch

name=det_rot_bposd_xzzx_zbias  
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation xzzx --ratio "$ratio" \
    --sizes "6,9,12,15" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "0.5,1" --prob "0.01:0.09:0.005"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation xzzx --ratio "$ratio" \
    --sizes "6,9,12,15" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "3" --prob "0.05:0.12:0.005"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation xzzx --ratio "$ratio" \
    --sizes "6,9,12,15" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "10" --prob "0.08:0.18:0.005"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation xzzx --ratio "$ratio" \
    --sizes "6,9,12,15" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "30" --prob "0.29:0.35:0.005"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation xzzx --ratio "$ratio" \
    --sizes "6,9,12,15" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "70" --prob "0.31:0.38:0.005"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation xzzx --ratio "$ratio" \
    --sizes "6,9,12,15" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "100" --prob "0.31:0.38:0.005"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 10 --queue $queue \
    --wall_time "$wall_time" --trials 10000 --split 1 $sbatch_dir/$name.sbatch

name=det_rot_sweepmatch_undef_zbias   
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation none --ratio "$ratio" \
    --sizes "2,4,6,8" --decoder RotatedSweepMatchDecoder --bias Z \
    --eta "0.5,1" --prob "0.01:0.10:0.005"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation none --ratio "$ratio" \
    --sizes "2,4,6,8" --decoder RotatedSweepMatchDecoder --bias Z \
    --eta "3" --prob "0.10:0.17:0.005"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation none --ratio "$ratio" \
    --sizes "2,4,6,8" --decoder RotatedSweepMatchDecoder --bias Z \
    --eta "10,30,100,inf" --prob "0.10:0.16:0.005"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 5 --queue $queue \
    --wall_time "$wall_time" --trials 10000 --split 6 $sbatch_dir/$name.sbatch

name=det_rot_sweepmatch_xzzx_zbias
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation xzzx --ratio "$ratio" \
    --sizes "2,4,6,8" --decoder DeformedRotatedSweepMatchDecoder --bias Z \
    --eta "0.5,1" --prob "0.01:0.09:0.005"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation xzzx --ratio "$ratio" \
    --sizes "2,4,6,8" --decoder DeformedRotatedSweepMatchDecoder --bias Z \
    --eta "3" --prob "0.05:0.11:0.005"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation xzzx --ratio "$ratio" \
    --sizes "2,4,6,8" --decoder DeformedRotatedSweepMatchDecoder --bias Z \
    --eta "10" --prob "0.11:0.17:0.005"
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation xzzx --ratio "$ratio" \
    --sizes "2,4,6,8" --decoder DeformedRotatedSweepMatchDecoder --bias Z \
    --eta "30,100,inf" --prob "0.14:0.24:0.005"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 5 --queue $queue \
    --wall_time "$wall_time" --trials 10000 --split 6 $sbatch_dir/$name.sbatch

# Rough runs using InfZ Optimal decoder on rotated code.
name=rot_infzopt_xzzx_zbias
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation xzzx --ratio "$ratio" \
    --sizes "2,4,6,8,10" --decoder RotatedInfiniteZBiasDecoder --bias Z \
    --eta "inf" --prob "0:1:0.01"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 12 --queue $queue \
    --wall_time "$wall_time" --trials 2000 --split 10 $sbatch_dir/$name.sbatch

# Rough runs using SweepMatch decoder on unrotated code.
name=unrot_sweepmatch_undef_zbias
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation none --ratio "$ratio" \
    --sizes "4,6,8,10" --decoder SweepMatchDecoder --bias Z \
    --eta "0.5,1,10,100,inf" --prob "0:0.4:0.02"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 6 --queue $queue \
    --wall_time "$wall_time" --trials 1000 --split 10 $sbatch_dir/$name.sbatch

name=unrot_sweepmatch_xzzx_zbias
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation xzzx --ratio "$ratio" \
    --sizes "4,6,8,10" --decoder DeformedSweepMatchDecoder --bias Z \
    --eta "0.5,1,10,100,inf" --prob "0:0.4:0.02"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 6 --queue $queue \
    --wall_time "$wall_time" --trials 1000 --split 10 $sbatch_dir/$name.sbatch

# Rough runs using BPOSD decoder on toric code
name=unrot_bposd_xzzx_zbias
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation xzzx --ratio "$ratio" \
    --sizes "5,9,13,17,21" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "30,100,inf" --prob "0:0.5:0.02"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 100 --queue $queue \
    --wall_time "$wall_time" --trials 10000 --split 10 $sbatch_dir/$name.sbatch

name=unrot_bposd_undef_zbias
panqec generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation none --ratio "$ratio" \
    --sizes "5,9,13,17,21" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "30,100,inf" --prob "0:0.5:0.02"
panqec pi-sbatch --data_dir "$paper_dir/$name" --n_array 100 --queue $queue \
    --wall_time "$wall_time" --trials 10000 --split 10 $sbatch_dir/$name.sbatch
'
