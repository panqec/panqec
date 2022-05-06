paper_dir=temp/paper
mkdir -p "$paper_dir"
sbatch_dir=temp/paper/sbatch
mkdir -p "$sbatch_dir"
mkdir -p temp/paper/share
qos=dpart

# ============== Undeformed ==============

name=rough_xcube_bposd_undef_zbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
# sizes="9,13,17,21"
sizes="7,9,11,13"
wall_time="47:59:00"
memory="20G"

# estimated p_th = 0.051
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "0.5" --prob "0.03:0.07:0.005"

# estimated p_th unknown
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "3" --prob "0.04:0.10:0.005"

# estimated p_th = 0.0988
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "10" --prob "0.08:0.12:0.005"

# estimated p_th = 0.0962
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "30" --prob "0.08:0.12:0.005"

# estimated p_th = 0.0951
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "100" --prob "0.08:0.12:0.005"

# estimated p_th unknown
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "inf" --prob "0.08:0.12:0.005"

panqec umiacs-sbatch --data_dir "$paper_dir/$name" --n_array 59 \
    --memory "$memory" --qos "$qos" \
    --wall_time "$wall_time" --trials 10000 --split 16 $sbatch_dir/$name.sbatch


# ============== Deformed ==============

name=rough_xcube_bposd_xzzx_zbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
# sizes="9,13,17,21"
sizes="7,9,11,13"
wall_time="47:59:00"
memory="20G"

# estimated p_th unknown
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class DeformedXZZXErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "0.5" --prob "0.02:0.07:0.005"

# estimated p_th unknown
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class DeformedXZZXErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "3" --prob "0.01:0.2:0.01"

# estimated p_th unknown
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class DeformedXZZXErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "10" --prob "0.02:0.3:0.02"

# estimated p_th unknown
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class DeformedXZZXErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "30" --prob "0.03:0.50:0.03"

# estimated p_th unknown
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class DeformedXZZXErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "100" --prob "0.03:0.50:0.03"

# estimated p_th unknown
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class DeformedXZZXErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "inf" --prob "0.03:0.50:0.03"

panqec umiacs-sbatch --data_dir "$paper_dir/$name" --n_array 8 \
    --memory "$memory" --qos "$qos" \
    --wall_time "$wall_time" --trials 1000 --split 1 $sbatch_dir/$name.sbatch
