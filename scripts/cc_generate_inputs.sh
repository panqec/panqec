paper_dir=temp/paper
mkdir -p "$paper_dir"
sbatch_dir=temp/paper/sbatch
mkdir -p "$sbatch_dir"

wall_time="23:00:00"
memory="16GB"

# Rough runs using BPOSD decoder on toric code
name=unrot_bposd_xzzx_zbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation xzzx --ratio equal  \
    --sizes "5,9,13,17,21" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "30,100,inf" --prob "0:0.5:0.02"
bn3d cc-sbatch --data_dir "$paper_dir/$name" --n_array 100 \
    --wall_time "$wall_time" --trials 10000 --split 10 $sbatch_dir/$name.sbatch

name=unrot_bposd_undef_zbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
bn3d generate-input -i "$paper_dir/$name/inputs" \
    --lattice kitaev --boundary toric --deformation none --ratio equal \
    --sizes "5,9,13,17,21" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "30,100,inf" --prob "0:0.5:0.02"
bn3d cc-sbatch --data_dir "$paper_dir/$name" --n_array 100 \
    --wall_time "$wall_time" --trials 10000 --split 10 $sbatch_dir/$name.sbatch
