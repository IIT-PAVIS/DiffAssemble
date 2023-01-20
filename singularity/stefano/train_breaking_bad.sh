#!/bin/bash
cd /home/sfiorini/Positional_Puzzle/singularity # dove sono gli script

TIME='48:00:00'
dataset='wikiart'
puzzles='6 8 10 12'
steps=1
gpus=4
cpus=8
batch_size=8
sampling='DDIM'
inference_ratio=20
degree=0
max_epochs=1500

#path='Puzzle-Diff/dtlvbe4z/checkpoints/epoch=164-step=66165.ckpt'
#wandb_id='dtlvbe4z'

NAME="Diff-${dataset}-${steps}"
ARGS="-inference_ratio $inference_ratio -sampling $sampling -gpus $gpus -batch_size $batch_size -steps $steps -num_workers $cpus --noise_weight 0 --predict_xstart True  --backbone vn_dgcnn"
#--inf_fully True --degree $i --architecture exophormer --virt_nodes 8 --checkpoint_path $path --wandb_id $wandb_id"

echo $NAME
echo ""
echo $ARGS
qsub -l select=1:ngpus=$gpus:ncpus=$cpus -v pyfile=puzzle_diff/train_3d.py,args="$ARGS" -N "$NAME"  pbs_args_franklin2.sh

