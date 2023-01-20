#!/bin/bash


# Request 1 chunk of 20 cores and 4 gpus.
#PBS -l walltime=24:00:00

# Merge the error and output streams into a single file.
#PBS -j oe


# Specify the job name.


# Use the gpu nodes.
#PBS -q gpu


cd /home/gscarpellini/Positional_Puzzle
# Run the program.

module load go-1.19.4/apptainer-1.1.8 

#module load openmpi/4.0.5/gcc7-ib


echo "$pyfile"

singularity run --nv -B ./:/app -B /work/gscarpellini:/work/gscarpellini singularity/singularity.sif bash /app/singularity/gianscarpe/run_Script_args.sh $pyfile "$args" #$1
