paper_dir=temp/paper
mkdir -p "$paper_dir"
sbatch_dir=temp/paper/sbatch
mkdir -p "$sbatch_dir"
mkdir -p temp/paper/share

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

name=det_rhombic_bposd_undef_xbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="10,12,14,18"
wall_time="14-00:00"
memory="160GB"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "0.5" --prob "0.010:0.020:0.001"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "3" --prob "0.036:0.046:0.001"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "10" --prob "0.103:0.123:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "15" --prob "0.148:0.168:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "20" --prob "0.205:0.225:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "25" --prob "0.255:0.275:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "30" --prob "0.290:0.310:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "100,inf" --prob "0.284:0.304:0.002"
bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 55 --memory "$memory" \
    --wall_time "$wall_time" --trials 10000 --split 20 $sbatch_dir/$name.sbatch

name=det_rhombic_bposd_xzzx_xbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="10,12,14,18"
wall_time="14-00:00"
memory="160GB"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "0.5" --prob "0.010:0.020:0.001"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "3" --prob "0.025:0.035:0.001"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "10" --prob "0.05:0.07:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "30" --prob "0.09:0.13:0.004"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "100" --prob "0.185:0.205:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "inf" --prob "0.388:0.408:0.002"
bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 35 --memory "$memory" \
    --wall_time "$wall_time" --trials 10000 --split 8 $sbatch_dir/$name.sbatch
