paper_dir=temp/paper
mkdir -p "$paper_dir"
sbatch_dir=temp/paper/qsub
mkdir -p "$qsub_dir"
mkdir -p temp/paper/share

name=det_rhombic_bposd_undef_xbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="4,6"
wall_time="14-00:00"
memory="2GB"
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
    --wall_time "$wall_time" --trials 10000 --split 20 $sbatch_dir/$name.qsub

name=det_rhombic_bposd_xzzx_xbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="4,6"
wall_time="14-00:00"
memory="2GB"
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
bn3d generate-qsub --data_dir "$paper_dir/$name" --n_array 35 --memory "$memory" \
    --wall_time "$wall_time" --trials 100 --split 8 $sbatch_dir/$name.qsub
