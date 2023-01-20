#!/bin/bash
cd /home/gscarpellini/Positional_Puzzle/singularity/gianscarpe

dataset='celeba'
puzzles='6 8 10 12 14 16 18 20'
steps=300
gpus=1
cpus=4
batch_size=8
sampling='DDIM'
inference_ratio=10
degree=60

NAME="Diff-${dataset}-${steps}-degree-${degree}"
ARGS="-dataset $dataset -puzzle_sizes $puzzles -inference_ratio $inference_ratio -sampling $sampling -gpus $gpus --predict_xstart True -batch_size $batch_size -steps $steps -num_workers $cpus --noise_weight 0 --virt_nodes 8 --rotation True --degree $degree% --backbone resnet18equiv --architecture exophormer --acc_grad 0"

echo $NAME
echo ""
echo $ARGS
qsub -q gpu_a100 -l select=1:ngpus=$gpus:ncpus=$cpus -v pyfile=puzzle_diff/train_script.py,args="$ARGS" -N "$NAME"  pbs_args.sh
