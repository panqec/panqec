# estimated p_th = 0.0482
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
    --eta "3" --prob "0.10:0.18:0.01"

# estimated p_th = 0.1022
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "10" --prob "0.08:0.12:0.005"

# estimated p_th = 0.0988
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "30" --prob "0.08:0.12:0.005"

# estimated p_th = 0.0975
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "100" --prob "0.08:0.12:0.005"

# estimated p_th = 0.0973
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "inf" --prob "0.08:0.12:0.005"
