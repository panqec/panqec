paper_dir=temp/paper
mkdir -p "$paper_dir"
sbatch_dir=temp/paper/sbatch
mkdir -p "$sbatch_dir"

wall_time="0-23:50"
memory="90GB"

name=rot_bposd_xzzx_zbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation xzzx --ratio equal  \
    --sizes "4,6,8,10" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "0.5,1,3,10,30,100,inf" --prob "0:0.5:0.02"
bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 50 --memory "$memory" \
    --wall_time "$wall_time" --trials 2000 --split 2 $sbatch_dir/$name.sbatch

name=rot_bposd_undef_zbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --lattice rotated --boundary planar --deformation none --ratio equal \
    --sizes "4,6,8,10" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "0.5,1,3,10,30,100,inf" --prob "0:0.5:0.02"
bn3d nist-sbatch --data_dir "$paper_dir/$name" --n_array 50 --memory "$memory" \
    --wall_time "$wall_time" --trials 2000 --split 2 $sbatch_dir/$name.sbatch
