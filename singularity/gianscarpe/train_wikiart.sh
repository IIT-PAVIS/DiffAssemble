#!/bin/bash
cd /home/gscarpellini/PuzzleDiffusion-GNN/singularity/gianscarpe

dataset='wikiart'
puzzles='12'
steps=100
gpus=2
cpus=1
batch_size=16
sampling='DDIM'
inference_ratio=10

NAME="Diff-${dataset}-${puzzles}-${steps}"
ARGS="-dataset $dataset -puzzle_sizes $puzzles -inference_ratio $inference_ratio -sampling $sampling -gpus $gpus -batch_size $batch_size -steps $steps -num_workers $cpus  --classifier_free_prob 0.1 --classifier_free_w 0.2 --noise_weight 0.0"

echo $NAME
echo ""
echo $ARGS
qsub -l select=1:ngpus=$gpus:ncpus=$cpus -v pyfile=puzzle_diff/train_script.py,args="$ARGS" -N "$NAME"  pbs_args.sh
