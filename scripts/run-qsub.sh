#!/bin/bash

qsub -t 1-`ls $HOME/bn3d/results/input/ | wc -l` $HOME/bn3d/scripts/qsub.job