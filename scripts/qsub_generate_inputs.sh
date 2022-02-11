paper_dir=temp/paper
mkdir -p "$paper_dir"
qsub_dir=temp/paper/qsub
mkdir -p "$qsub_dir"
mkdir -p temp/paper/share

name=det_xcube_bposd_undef_zbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="4,6"
wall_time="14:00:00"
memory="2G"

bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "0.5" --prob "0.001:0.5:0.01"

bn3d generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class PauliErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "100" --prob "0.001:0.5:0.01"

bn3d generate-qsub --data_dir "$paper_dir/$name" --n_array 23 --memory "$memory" \
    --wall_time "$wall_time" --trials 100 --split 10 "$qsub_dir/$name.qsub"