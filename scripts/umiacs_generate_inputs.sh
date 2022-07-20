paper_dir=temp/paper
mkdir -p "$paper_dir"
sbatch_dir=temp/paper/sbatch
mkdir -p "$sbatch_dir"
mkdir -p temp/paper/share
qos=dpart

# =============== Test Run ================
name=test_run
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="3,4,5"
wall_time="0:02:00"
memory="20G"

panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class Toric2DCode --noise_class DeformedXZZXErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias X \
    --eta "10,30" --prob "0.0:0.5:0.07"

nfiles=$(ls $paper_dir/$name/inputs | wc -l)
echo "$nfiles input files created"
panqec umiacs-sbatch --data_dir "$paper_dir/$name" --n_array 3 \
    --memory "$memory" --qos "$qos" \
    --wall_time "$wall_time" --trials 300 --split auto $sbatch_dir/$name.sbatch

# =============== Rhombic Undeformed ================
name=umiacs_rhombic_bposd_undef_xbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="10,14,18,22"
# sizes="8,10,12,14"
wall_time="24:00:00"
memory="20G"

source scripts/rhombic_undef.sh

nfiles=$(ls $paper_dir/$name/inputs | wc -l)
echo "$nfiles input files created"
panqec umiacs-sbatch --data_dir "$paper_dir/$name" --n_array $nfiles \
    --memory "$memory" --qos "$qos" \
    --wall_time "$wall_time" --trials 10000 --split auto $sbatch_dir/$name.sbatch

# =============== Rhombic Deformed ================
name=umiacs_rhombic_bposd_xzzx_xbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="10,14,18,20"
# sizes="8,10,12,14"
wall_time="24:00:00"
memory="20G"

source scripts/rhombic_xzzx.sh

nfiles=$(ls $paper_dir/$name/inputs | wc -l)
echo "$nfiles input files created"
panqec umiacs-sbatch --data_dir "$paper_dir/$name" --n_array 8 \
    --memory "$memory" --qos "$qos" \
    --wall_time "$wall_time" --trials 250 --split auto $sbatch_dir/$name.sbatch

# ============== XCube Undeformed ==============
name=umiacs_xcube_bposd_undef_zbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="9,13,17,21"
# sizes="7,9,11,13"
wall_time="24:00:00"
memory="20G"

source scripts/xcube_undef.sh

nfiles=$(ls $paper_dir/$name/inputs | wc -l)
echo "$nfiles input files created"
panqec umiacs-sbatch --data_dir "$paper_dir/$name" --n_array $nfiles \
    --memory "$memory" --qos "$qos" \
    --wall_time "$wall_time" --trials 10000 --split auto $sbatch_dir/$name.sbatch

# ============== XCube Deformed ==============
name=umiacs_xcube_bposd_xzzx_zbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="9,13,17,21"
# sizes="7,9,11,13"
wall_time="24:00:00"
memory="20G"

source scripts/xcube_xzzx.sh

nfiles=$(ls $paper_dir/$name/inputs | wc -l)
echo "$nfiles input files created"
panqec umiacs-sbatch --data_dir "$paper_dir/$name" --n_array $nfiles \
    --memory "$memory" --qos "$qos" \
    --wall_time "$wall_time" --trials 10000 --split auto $sbatch_dir/$name.sbatch

# ============== XCube Deformed Inf Only ==============
common_name=umiacs_xcube_bposd_xzzx_zbias_inf
name=${common_name}_temp
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="7,9,11,13"

panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class DeformedXZZXErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "inf" --prob "0.10:0.20:0.02"

nfiles=$(ls $paper_dir/$name/inputs | wc -l)
echo "$nfiles input files created"

for repeat in $(seq 1 8); do
    name=${common_name}_${repeat}
    mkdir -p $paper_dir/$name
    rm -rf $paper_dir/$name/inputs
    rm -rf $paper_dir/$name/logs
    cp -R $paper_dir/${common_name}_temp/inputs $paper_dir/$name/inputs
    panqec umiacs-sbatch --data_dir "$paper_dir/$name" --n_array $nfiles \
        --memory "$memory" --qos "$qos" \
        --wall_time "$wall_time" --trials 125 --split auto $sbatch_dir/$name.sbatch
done

rm -rf $paper_dir/${common_name}_temp
