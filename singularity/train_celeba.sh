#!/bin/bash
cd /work/sfiorini/Positional_Puzzle/singularity # dove sono gli script

dataset='wikiart'
puzzles='6 8 10 12'
steps=600
gpus=4
cpus=6
batch_size=32
sampling='DDIM'
checkpoint_path='../Puzzle-Diff/etzpjrkl/checkpoints/epoch=379-step=152380.ckpt'
inference_ratio=1


NAME="Diff-${dataset}-${steps}"
ARGS="-dataset $dataset -puzzle_sizes $puzzles -inference_ratio $inference_ratio -sampling $sampling -gpus $gpus -batch_size $batch_size -steps $steps -num_workers $cpus --noise_weight 0 --predict_xstart True --rotation True --checkpoint_path $checkpoint_path"

 

echo $NAME
echo ""
echo $ARGS
qsub -l select=1:ngpus=$gpus:ncpus=$cpus -v pyfile=puzzle_diff/train_script.py,args="$ARGS" -N "$NAME"  pbs_args.sh
