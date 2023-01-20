
#!/bin/bash
cd /home/gscarpellini/PuzzleDiffusion-GNN/singularity/gianscarpe

dataset='celeba'
puzzles='6'
steps=1
gpus=3
cpus=6
batch_size=8
sampling='DDIM'
inference_ratio=1

NAME="Diff-${dataset}-${puzzles}-${steps}"
ARGS="-dataset $dataset -puzzle_sizes $puzzles -inference_ratio $inference_ratio -sampling $sampling -gpus $gpus -batch_size $batch_size -steps $steps -num_workers $cpus --noise_weight 0 --predict_xstart True --discrete True"

echo $NAME
echo ""
echo $ARGS
qsub -l select=1:ngpus=$gpus:ncpus=$cpus -v pyfile=puzzle_diff/train_script.py,args="$ARGS" -N "$NAME"  pbs_args.sh
