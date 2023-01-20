#!/bin/bash


# Request 1 chunk of 20 cores and 4 gpus.

# Merge the error and output streams into a single file.
#PBS -j oe


# Specify the job name.


# Use the gpu nodes.
#PBS -q gpu_a100


cd /work/sfiorini/Positional_Puzzle
# Run the program.
#module load cuda/11.6
#module load openmpi/4.0.5/gcc7-ib
module load go-1.19.4/apptainer-1.1.8

echo "$pyfile"

# singularity exec -B /work/fgiuliari/Pointnet_Scannet -B /work/fgiuliari/Scannet-bbox:/work/fgiuliari/Pointnet_Scannet/Scannet --nv env.sif python train_pointnet.py --batch_size 20
singularity run --nv -B ./:/app -B /work/sfiorini:/work/sfiorini singularity/build/diffPYG.sif bash /app/singularity/run_Script_args.sh $pyfile "$args" #$1
