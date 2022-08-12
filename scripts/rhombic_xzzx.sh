# estimated p_th = 0.015
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "0.5" --prob "0.010:0.018:0.001"

# estimated p_th = 0.0298
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "3" --prob "0.0:0.05:0.005"

# estimated p_th = 0.0517
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "10" --prob "0.0:0.10:0.01"

# estimated p_th = 0.1155
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "30" --prob "0.06:0.16:0.01"

# estimated p_th = 0.18
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "100" --prob "0.13:0.23:0.01"

# estimated p_th = 0.50
panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class RhombicCode --noise_class DeformedRhombicErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "inf" --prob "0.03:0.50:0.03"
