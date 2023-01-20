#!/bin/bash
cd /work/fgiuliari/PuzzleDiffusion-GNN/singularity

NAME='Diff-GNN'
ARGS='-gpus 2 -batch_size 24 -steps 400 -num_workers 10'


qsub -v pyfile=puzzle_diff/train_diff.py,args="$ARGS" -N "$NAME"  pbs_args.sh
