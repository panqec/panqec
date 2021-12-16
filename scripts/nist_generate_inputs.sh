paper_dir=temp/paper
mkdir -p "$paper_dir"
sbatch_dir=temp/paper/sbatch
mkdir -p "$sbatch_dir"

wall_time="0-23:50"
memory="90GB"

# name=rot_bposd_xzzx_zbias
# rm -rf $paper_dir/$name/inputs
# rm -rf $paper_dir/$name/logs
# bn3d generate-input -i "$paper_dir/$name/inputs" \
    # --lattice rotated --boundary planar --deformation xzzx --ratio equal  \
    # --sizes "4,6,8,10" --decoder BeliefPropagationOSDDecoder --bias Z \
    # --eta "0.5,1,3,10,30,100,inf" --prob "0:0.5:0.02"
# bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 50 --memory "$memory" \
    # --wall_time "$wall_time" --trials 2000 --split 2 $sbatch_dir/$name.sbatch

# name=rot_bposd_undef_zbias
# rm -rf $paper_dir/$name/inputs
# rm -rf $paper_dir/$name/logs
# bn3d generate-input -i "$paper_dir/$name/inputs" \
    # --lattice rotated --boundary planar --deformation none --ratio equal \
    # --sizes "4,6,8,10" --decoder BeliefPropagationOSDDecoder --bias Z \
    # --eta "0.5,1,3,10,30,100,inf" --prob "0:0.5:0.02"
# bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 50 --memory "$memory" \
    # --wall_time "$wall_time" --trials 2000 --split 2 $sbatch_dir/$name.sbatch


name=det_rhombic_bposd_undef_xbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "9,13,17,21" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "0.5,1" --prob "0.006:0.025:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "9,13,17,21" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "3" --prob "0.02:0.06:0.005"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "9,13,17,21" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "10" --prob "0.08:0.12:0.005"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "9,13,17,21" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "15" --prob "0.10:0.20:0.01"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "9,13,17,21" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "20" --prob "0.16:0.26:0.01"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "9,13,17,21" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "25" --prob "0.21:0.32:0.01"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "9,13,17,21" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "30,100,inf" --prob "0.28:0.32:0.005"
bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 104 --memory "$memory" \
    --wall_time "$wall_time" --trials 2000 --split 2 $sbatch_dir/$name.sbatch

name=det_rhombic_bposd_xzzx_xbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "9,13,17,21" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "0.5" --prob "0.01:0.02:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "9,13,17,21" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "1" --prob "0.015:0.025:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "9,13,17,21" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "3" --prob "0.02:0.04:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "9,13,17,21" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "10" --prob "0.05:0.07:0.002"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "9,13,17,21" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "30" --prob "0.09:0.13:0.005"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "9,13,17,21" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "100" --prob "0.18:0.22:0.005"
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "6,8,10,12" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "inf" --prob "0.38:0.42:0.005"
bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 65 --memory "$memory" \
    --wall_time "$wall_time" --trials 2000 --split 2 $sbatch_dir/$name.sbatch
