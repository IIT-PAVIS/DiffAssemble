#!/bin/bash


# Request 1 chunk of 20 cores and 4 gpus.

# Merge the error and output streams into a single file.
#PBS -j oe


# Specify the job name.


# Use the gpu nodes.
#PBS -q workq


cd /home/sfiorini/Positional_Puzzle
# Run the program.
#module load cuda/11.6
#module load openmpi/4.0.5/gcc7-ib
#module load go-1.19.4/apptainer-1.1.8
module load cuda/11.7 mpi/openmpi/4.1.4/gcc8-ib singularity/3.10

echo "$pyfile"

# singularity exec -B /work/fgiuliari/Pointnet_Scannet -B /work/fgiuliari/Scannet-bbox:/work/fgiuliari/Pointnet_Scannet/Scannet --nv env.sif python train_pointnet.py --batch_size 20
singularity run --nv -B ./:/app -B /home/sfiorini:/home/sfiorini singularity/build/singularity.sif bash /app/singularity/run_Script_args.sh $pyfile "$args" #$1

