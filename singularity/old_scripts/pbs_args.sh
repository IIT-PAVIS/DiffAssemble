#!/bin/bash


# Request 1 chunk of 20 cores and 4 gpus.
#PBS -l select=1:ncpus=8:ngpus=3
#PBS -l walltime=24:00:00

# Merge the error and output streams into a single file.
#PBS -j oe


# Specify the job name.


# Use the gpu nodes.
#PBS -q gpu


cd /work/fgiuliari/PuzzleDiffusion-GNN
# Run the program.
module load cuda/11.6
#module load openmpi/4.0.5/gcc7-ib


echo "$pyfile"

# singularity exec -B /work/fgiuliari/Pointnet_Scannet -B /work/fgiuliari/Scannet-bbox:/work/fgiuliari/Pointnet_Scannet/Scannet --nv env.sif python train_pointnet.py --batch_size 20
singularity run --nv -B ./:/app singularity/build/diffPYG.sif bash /app/singularity/run_Script_args.sh $pyfile "$args" #$1
