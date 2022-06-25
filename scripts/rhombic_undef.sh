# estimated p_th = 0.014
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "0.5" --prob "0.010:0.018:0.001"

# estimated p_th = 0.0377
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "3" --prob "0.025:0.045:0.002"

# estimated p_th = 0.1030
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "10" --prob "0.08:0.12:0.004"

# estimated p_th = 0.157
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "15" --prob "0.13:0.18:0.005"

# estimated p_th = 0.1977
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "20" --prob "0.18:0.22:0.004"

# estimated p_th = 0.2919
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "25" --prob "0.27:0.30:0.004"

# estimated p_th = 0.2919
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "30" --prob "0.27:0.30:0.004"

panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "100,inf" --prob "0.27:0.30:0.004"
