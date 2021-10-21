paper_dir=temp/paper
mkdir -p "$paper_dir"
sbatch_dir=temp/paper/sbatch
mkdir -p "$sbatch_dir"


function setup_dir () {
	local name=$1 rotation=$2 decoder=$3 deformation=$4 bias=$5
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
            -s 2,4,6,8 --decoder $decoder --bias $bias

    walltime='0-03:00'
    trials=1000
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

setup_dir rot_bposd_undef_xbias rotated BeliefPropagationOSDDecoder none X
setup_dir rot_bposd_undef_zbias rotated BeliefPropagationOSDDecoder none Z
setup_dir rot_bposd_xzzx_xbias rotated BeliefPropagationOSDDecoder xzzx X
setup_dir rot_bposd_xzzx_zbias rotated BeliefPropagationOSDDecoder xzzx Z
setup_dir rot_sweepmatch_undef_xbias rotated RotatedSweepMatchDecoder none X
setup_dir rot_sweepmatch_undef_zbias rotated RotatedSweepMatchDecoder none Z
setup_dir rot_sweepmatch_xy_xbias rotated RotatedSweepMatchDecoder xy X
setup_dir rot_sweepmatch_xy_zbias rotated RotatedSweepMatchDecoder xy Z

setup_dir unrot_bposd_undef_xbias kitaev BeliefPropagationOSDDecoder none X
setup_dir unrot_bposd_undef_zbias kitaev BeliefPropagationOSDDecoder none Z
setup_dir unrot_bposd_xzzx_xbias kitaev BeliefPropagationOSDDecoder xzzx X
setup_dir unrot_bposd_xzzx_zbias kitaev BeliefPropagationOSDDecoder xzzx Z
setup_dir unrot_sweepmatch_undef_xbias kitaev SweepMatchDecoder none X
setup_dir unrot_sweepmatch_undef_zbias kitaev SweepMatchDecoder none Z
setup_dir unrot_sweepmatch_xy_xbias kitaev SweepMatchDecoder xy X
setup_dir unrot_sweepmatch_xy_zbias kitaev SweepMatchDecoder xy Z
