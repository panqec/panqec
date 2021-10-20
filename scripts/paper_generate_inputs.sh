paper_dir=temp/paper
mkdir -p "$paper_dir"


function setup_dir () {
	local name=$1 decoder=$2 deformation=$3 bias=$4
	input_dir="$paper_dir/$name/inputs"
	mkdir -p $input_dir
	bn3d generate-input -i $input_dir \
			-l rotated -b planar -d $deformation -r equal \
			-s 2,4,6,8 --decoder $decoder --bias $bias
}

setup_dir bposd_undef_xbias BeliefPropagationOSDDecoder none X
setup_dir bposd_undef_zbias BeliefPropagationOSDDecoder none Z
setup_dir bposd_xzzx_xbias BeliefPropagationOSDDecoder xzzx X
setup_dir bposd_xzzx_zbias BeliefPropagationOSDDecoder xzzx Z
setup_dir sweepmatch_undef_xbias RotatedSweepMatchDecoder none X
setup_dir sweepmatch_undef_zbias RotatedSweepMatchDecoder none Z
setup_dir sweepmatch_xy_xbias RotatedSweepMatchDecoder xy X
setup_dir sweepmatch_xy_zbias RotatedSweepMatchDecoder xy Z
