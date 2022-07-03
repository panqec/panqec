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
    --wall_time "$wall_time" --trials 3000 --split auto $sbatch_dir/$name.sbatch

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
name=umiacs_xcube_bposd_xzzx_zbias_inf
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="9,13,15"
# sizes="7,9,11,13"
wall_time="00:05:00"
memory="20G"

panqec generate-input -i "$paper_dir/$name/inputs" \
    --code_class XCubeCode --noise_class DeformedXZZXErrorModel \
    --ratio equal \
    --sizes "$sizes" --decoder BeliefPropagationOSDDecoder --bias Z \
    --eta "inf" --prob "0.05:0.50:0.05"

nfiles=$(ls $paper_dir/$name/inputs | wc -l)
echo "$nfiles input files created"
panqec umiacs-sbatch --data_dir "$paper_dir/$name" --n_array $nfiles \
    --memory "$memory" --qos "$qos" \
    --wall_time "$wall_time" --trials 1000 --split auto $sbatch_dir/$name.sbatch
