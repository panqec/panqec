#!/bin/bash

qsub -t 1-`ls $HOME/panqec/results/input/ | wc -l` $HOME/panqec/scripts/qsub.job