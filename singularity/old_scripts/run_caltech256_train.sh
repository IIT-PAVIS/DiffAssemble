#!/bin/bash
cd /work/fgiuliari/PuzzleDiffusion-GNN/singularity

NAME='Diff-GNN'
ARGS='-gpus 4 -batch_size 4 -steps 400'


qsub -v pyfile=puzzle_diff/train_diff_caltech256.py,args="$ARGS" -N "$NAME"  pbs_args.sh
