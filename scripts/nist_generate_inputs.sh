paper_dir=temp/paper
mkdir -p "$paper_dir"
sbatch_dir=temp/paper/sbatch
mkdir -p "$sbatch_dir"

wall_time="4-15:00"
memory="93GB"

name=det_rot_bposd_xzzx_zbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="4,6,8,10"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation xzzx --ratio equal  \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "0.5" --prob "0.055:0.075:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation xzzx --ratio equal  \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "1" --prob "0.050:0.070:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation xzzx --ratio equal  \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "3" --prob "0.060:0.100:0.004"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation xzzx --ratio equal  \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "10" --prob "0.011:0.015:0.004"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation xzzx --ratio equal  \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "30" --prob "0.326:0.336:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation xzzx --ratio equal  \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "100" --prob "0.36:0.38:0.004"
bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 51 --memory "$memory" \
    --wall_time "$wall_time" --trials 5000 --split 2 $sbatch_dir/$name.sbatch

name=det_rot_bposd_undef_zbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation none --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "0.5" --prob "0.055:0.075:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation none --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "0.5" --prob "0.066:0.086:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation none --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "3" --prob "0.120:0.140:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation none --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "10" --prob "0.218:0.238:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation none --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "30" --prob "0.208:0.228:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation none --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "100" --prob "0.206:0.226:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation none --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "inf" --prob "0.206:0.226:0.002"
bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 81 --memory "$memory" \
    --wall_time "$wall_time" --trials 5000 --split 2 $sbatch_dir/$name.sbatch


name=det_rhombic_bposd_undef_xbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="9,11,13,17"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "0.5,1" --prob "0.006:0.025:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "3" --prob "0.02:0.06:0.005"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "10" --prob "0.08:0.12:0.005"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "15" --prob "0.10:0.20:0.01"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "20" --prob "0.16:0.26:0.01"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "25" --prob "0.21:0.32:0.01"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "30,100,inf" --prob "0.28:0.32:0.005"
bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 104 --memory "$memory" \
    --wall_time "$wall_time" --trials 5000 --split 2 $sbatch_dir/$name.sbatch

name=det_rhombic_bposd_xzzx_xbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "0.5" --prob "0.01:0.02:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "1" --prob "0.015:0.025:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "3" --prob "0.02:0.04:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "10" --prob "0.05:0.07:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "30" --prob "0.09:0.13:0.005"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "100" --prob "0.18:0.22:0.005"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "inf" --prob "0.38:0.42:0.005"
bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 65 --memory "$memory" \
    --wall_time "$wall_time" --trials 5000 --split 2 $sbatch_dir/$name.sbatch
