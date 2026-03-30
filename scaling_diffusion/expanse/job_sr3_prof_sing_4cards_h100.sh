#!/usr/bin/env bash
#SBATCH --job-name="sr3_4cards_h100"
#SBATCH --output="out_sr3_4cards_h100.%j.%N.out"
#SBATCH --partition=nairr-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:4
#SBATCH --mem=32G
#SBATCH --account=sds279
#SBATCH -t 01:30:00


declare -xr SINGUALRITY_MODULE='singularitypro/4.1.2'

module purge
module load "${SINGUALRITY_MODULE}"
#module list
#printenv

WORKDIR=$HOME/models/sr3_scaling_feb26

date

#### We will run from the scratch directory local to the node
cd /scratch/$USER/job_$SLURM_JOBID

#time -p singularity exec --bind /home,/expanse,/scratch --nv /cm/shared/apps/containers/singularity/pytorch/pytorch_2.2.1-openmpi-4.1.6-mofed-5.8-2.0.3.0-cuda-12.1.1-ubuntu-22.04.4-x86_64-20240412.sif $WORKDIR/run_sr_1gpu.sh #python3 $SLURM_SUBMIT_DIR/main.py

time -p singularity exec --bind /home,/expanse,/scratch --nv /expanse/projects/qstore/installs/containers/singularity/Expanse-Air/pytorch/pytorch-nvcr-25.03.sif $WORKDIR/run_sr_4gpu.sh #python3 $SLURM_SUBMIT_DIR/main.py

#### List scratch directory
ls /scratch/$USER/job_$SLURM_JOBID
