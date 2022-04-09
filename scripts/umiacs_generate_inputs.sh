paper_dir=temp/paper
mkdir -p "$paper_dir"
sbatch_dir=temp/paper/sbatch
mkdir -p "$sbatch_dir"
mkdir -p temp/paper/share
qos=dpart

# ============== Undeformed ==============

name=det_xcube_bposd_undef_zbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="9,13,17,21"
wall_time="47:59:00"
memory="20G"

panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "0.5" --prob "0.05:0.15:0.005"

panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "3" --prob "0.08:0.18:0.005"

panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "10" --prob "0.07:0.17:0.005"

panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "30" --prob "0.07:0.17:0.005"

panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "100" --prob "0.07:0.17:0.005"

panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "inf" --prob "0.07:0.17:0.005"

panqec umiacs-sbatch --data_dir "$paper_dir/$name" --n_array 126 \
    --memory "$memory" --qos "$qos" \
    --wall_time "$wall_time" --trials 10000 --split 16 $sbatch_dir/$name.sbatch


# ============== Deformed ==============

name=det_xcube_bposd_xzzx_zbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="9,13,17,21"
wall_time="47:59:00"
memory="20G"

panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class DeformedXZZXErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "0.5" --prob "0.05:0.15:0.005"

panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class DeformedXZZXErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "3" --prob "0.06:0.15:0.005"

panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class DeformedXZZXErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "10" --prob "0.08:0.2:0.005"

panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class DeformedXZZXErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "30" --prob "0.1:0.25:0.005"

panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class DeformedXZZXErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "100" --prob "0.013:0.3:0.005"

panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class DeformedXZZXErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "inf" --prob "0.001:0.5:0.01"

panqec umiacs-sbatch --data_dir "$paper_dir/$name" --n_array 206 \
    --memory "$memory" --qos "$qos" \
    --wall_time "$wall_time" --trials 10000 --split 16 $sbatch_dir/$name.sbatch
