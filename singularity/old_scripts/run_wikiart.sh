#!/bin/bash
cd /work/fgiuliari/PuzzleDiffusion-GNN/singularity

NAME='Diff-GNN'
ARGS='-gpus 3 -batch_size 8 -steps 700 -num_workers 8'


qsub -v pyfile=puzzle_diff/train_wikiart.py,args="$ARGS" -N "$NAME"  pbs_args.sh
