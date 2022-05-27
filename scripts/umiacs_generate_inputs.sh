paper_dir=temp/paper
mkdir -p "$paper_dir"
sbatch_dir=temp/paper/sbatch
mkdir -p "$sbatch_dir"
mkdir -p temp/paper/share
qos=dpart

# =============== Rhombic Undeformed ================
name=det_rhombic_bposd_undef_xbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="10,14,18,22"
# sizes="8,10,12,14"
wall_time="1:30:00"
memory="20G"

source scripts/rhombic_undef.sh

nfiles=$(ls $paper_dir/$name/inputs | wc -l)
echo "$nfiles input files created"
panqec umiacs-sbatch --data_dir "$paper_dir/$name" --n_array $nfiles \
    --memory "$memory" --qos "$qos" \
    --wall_time "$wall_time" --trials 10000 --split 16 $sbatch_dir/$name.sbatch

# =============== Rhombic Deformed ================
name=det_rhombic_bposd_xzzx_xbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="10,14,18,22"
# sizes="8,10,12,14"
wall_time="1:30:00"
memory="20G"

source scripts/rhombic_xzzx.sh

nfiles=$(ls $paper_dir/$name/inputs | wc -l)
echo "$nfiles input files created"
panqec umiacs-sbatch --data_dir "$paper_dir/$name" --n_array $nfiles \
    --memory "$memory" --qos "$qos" \
    --wall_time "$wall_time" --trials 10000 --split 16 $sbatch_dir/$name.sbatch

# ============== XCube Undeformed ==============
name=rough_xcube_bposd_undef_zbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="9,13,17,21"
# sizes="7,9,11,13"
wall_time="0:10:00"
memory="20G"

source scripts/xcube_undef.sh

nfiles=$(ls $paper_dir/$name/inputs | wc -l)
echo "$nfiles input files created"
panqec umiacs-sbatch --data_dir "$paper_dir/$name" --n_array $nfiles \
    --memory "$memory" --qos "$qos" \
    --wall_time "$wall_time" --trials 10000 --split 16 $sbatch_dir/$name.sbatch

# ============== XCube Deformed ==============
name=rough_xcube_bposd_xzzx_zbias
rm -rf $paper_dir/$name/inputs
rm -rf $paper_dir/$name/logs
sizes="9,13,17,21"
# sizes="7,9,11,13"
wall_time="0:10:00"
memory="20G"

source scripts/xcube_xzzx.sh

nfiles=$(ls $paper_dir/$name/inputs | wc -l)
echo "$nfiles input files created"
panqec umiacs-sbatch --data_dir "$paper_dir/$name" --n_array $nfiles \
    --memory "$memory" --qos "$qos" \
    --wall_time "$wall_time" --trials 10000 --split 16 $sbatch_dir/$name.sbatch
