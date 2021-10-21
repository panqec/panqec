paper_dir=temp/paper
mkdir -p "$paper_dir"
sbatch_dir=temp/paper/sbatch
mkdir -p "$sbatch_dir"


setup_dir() {
    local name=$1 rotation=$2 decoder=$3 deformation=$4 bias=$5 eta=$6 \
            prob=$7 trials=$8
    echo "$name $rotation $decoder $deformation $bias"
    if [ $rotation = rotated ]; then
        boundary=planar
    else
        boundary=toric
    fi
    input_dir="$paper_dir/$name/inputs"
    mkdir -p $input_dir
    bn3d generate-input -i $input_dir \
        -l $rotation -b $boundary -d $deformation -r equal \
        -s 2,4,6,8 --decoder $decoder --bias $bias \
        --eta $eta --prob $prob

    walltime='0-03:00'
    narray=6
    queue=defq
    sbatch_file="$sbatch_dir/$name.sbatch"
    cp scripts/pi_template.sh "$sbatch_file"
    sed -i -- "s/\${TRIALS}/$trials/g" "$sbatch_file"
    sed -i -- "s/\${NAME}/$name/g" "$sbatch_file"
    sed -i -- "s/\${TIME}/$walltime/g" "$sbatch_file"
    sed -i -- "s/\${NARRAY}/$narray/g" "$sbatch_file"
    sed -i -- "s/\${QUEUE}/$queue/g" "$sbatch_file"
}

setup_dir sts_rot_bposd_undef_zbias rotated BeliefPropagationOSDDecoder none Z \
        "1,10,100,1000,inf" "0.05" 100000
setup_dir sts_rot_bposd_xzzx_zbias rotated BeliefPropagationOSDDecoder xzzx Z \
        "1,10,100,1000,inf" "0.05" 100000

setup_dir sts_rot_sweepmatch_undef_zbias rotated RotatedSweepMatchDecoder none Z \
        "1,10,100,1000,inf" "0.05" 100000
setup_dir sts_rot_sweepmatch_xy_zbias rotated RotatedSweepMatchDecoder xy Z \
        "1,10,100,1000,inf" "0.05" 100000

setup_dir rot_bposd_undef_zbias rotated BeliefPropagationOSDDecoder none Z \
        "0.5,1,3" "0:0.18:0.01" 10000
setup_dir rot_bposd_undef_zbias rotated BeliefPropagationOSDDecoder none Z \
        "10,30,100,inf" "0.18:0.27:0.01" 10000

setup_dir rot_bposd_xzzx_zbias rotated BeliefPropagationOSDDecoder xzzx Z \
        "0.5,1,3" "0:0.18:0.01" 10000
setup_dir rot_bposd_xzzx_zbias rotated BeliefPropagationOSDDecoder xzzx Z \
        "10,30,100,inf" "0:0.55:0.01" 10000

setup_dir rot_sweepmatch_undef_zbias rotated RotatedSweepMatchDecoder none Z \
        "0.5,1,3,10,30,100,inf" "0:0.55:0.01" 10000

setup_dir rot_sweepmatch_xy_zbias rotated RotatedSweepMatchDecoder xy Z \
        "0.5,1,3,10,30,100,inf" "0:0.55:0.01" 10000

: '
setup_dir rot_bposd_undef_xbias rotated BeliefPropagationOSDDecoder none X \
        "0.5,1,3,10,30,100,inf" "0:0.55:0.01"
setup_dir rot_bposd_undef_zbias rotated BeliefPropagationOSDDecoder none Z \
        "0.5,1,3,10,30,100,inf" "0:0.55:0.01"
setup_dir rot_bposd_xzzx_xbias rotated BeliefPropagationOSDDecoder xzzx X \
        "0.5,1,3,10,30,100,inf" "0:0.55:0.01"
setup_dir rot_bposd_xzzx_zbias rotated BeliefPropagationOSDDecoder xzzx Z \
        "0.5,1,3,10,30,100,inf" "0:0.55:0.01"
setup_dir rot_sweepmatch_undef_xbias rotated RotatedSweepMatchDecoder none X \
        "0.5,1,3,10,30,100,inf" "0:0.55:0.01"
setup_dir rot_sweepmatch_undef_zbias rotated RotatedSweepMatchDecoder none Z \
        "0.5,1,3,10,30,100,inf" "0:0.55:0.01"
setup_dir rot_sweepmatch_xy_xbias rotated RotatedSweepMatchDecoder xy X \
        "0.5,1,3,10,30,100,inf" "0:0.55:0.01"
setup_dir rot_sweepmatch_xy_zbias rotated RotatedSweepMatchDecoder xy Z \
        "0.5,1,3,10,30,100,inf" "0:0.55:0.01"

setup_dir unrot_bposd_undef_xbias kitaev BeliefPropagationOSDDecoder none X \
        "0.5,1,3,10,30,100,inf" "0:0.55:0.01"
setup_dir unrot_bposd_undef_zbias kitaev BeliefPropagationOSDDecoder none Z \
        "0.5,1,3,10,30,100,inf" "0:0.55:0.01"
setup_dir unrot_bposd_xzzx_xbias kitaev BeliefPropagationOSDDecoder xzzx X \
        "0.5,1,3,10,30,100,inf" "0:0.55:0.01"
setup_dir unrot_bposd_xzzx_zbias kitaev BeliefPropagationOSDDecoder xzzx Z \
        "0.5,1,3,10,30,100,inf" "0:0.55:0.01"
setup_dir unrot_sweepmatch_undef_xbias kitaev SweepMatchDecoder none X \
        "0.5,1,3,10,30,100,inf" "0:0.55:0.01"
setup_dir unrot_sweepmatch_undef_zbias kitaev SweepMatchDecoder none Z \
        "0.5,1,3,10,30,100,inf" "0:0.55:0.01"
setup_dir unrot_sweepmatch_xy_xbias kitaev SweepMatchDecoder xy X \
        "0.5,1,3,10,30,100,inf" "0:0.55:0.01"
setup_dir unrot_sweepmatch_xy_zbias kitaev SweepMatchDecoder xy Z \
        "0.5,1,3,10,30,100,inf" "0:0.55:0.01"
'
