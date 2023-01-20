
#!/bin/bash
cd /work/fgiuliari/PuzzleDiffusion-GNN/singularity/francesco

dataset='celeba'
puzzles='6'
steps=300
gpus=3
cpus=6
batch_size=32
sampling='DDIM'
inference_ratio=10

NAME="Diff-${dataset}-${puzzles}-${steps}"
ARGS="-dataset $dataset -puzzle_sizes $puzzles -inference_ratio $inference_ratio -sampling $sampling -gpus $gpus -batch_size $batch_size -steps $steps -num_workers $cpus"

echo $NAME
echo ""
echo $ARGS
qsub -l select=1:ngpus=$gpus:ncpus=$cpus -v pyfile=puzzle_diff/train_script_missing.py,args="$ARGS" -N "$NAME"  pbs_args.sh
