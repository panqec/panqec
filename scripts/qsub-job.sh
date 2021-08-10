#!/bin/bash -l

# Batch script to run a serial job under SGE.

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=0:5:0

# Request 1 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=1G

# Request 15 gigabyte of TMPDIR space (default is 10 GB - remove if cluster is diskless)
#$ -l tmpfs=1G

# Set up the job array.  In this instance we have requested 1000 tasks
# numbered 1 to 1000.
#$ -t 1-`ls results/input/ | wc -l`


# Set the name of the job.
#$ -N TestJob

# Automate transfer of output to Scratch from $TMPDIR.
#Local2Scratch

cd $TMPDIR

conda activate bn3d
filename="`ls` | awk '{print $3}'"
bn3d run --file /home/ucapacp/bn3d/results/input/$filename --trials 20 -o /home/ucapacp/Scratch/output