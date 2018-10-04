#!/usr/bin/env bash

#SBATCH -J preprocess
#SBATCH --mem-per-cpu 10000
##SBATCH --gres=gpu:p100:1
##SBATCH -p gpu
#SBATCH -t 1:00:00

#module purge
#module load intelconda/python3.6-2018.3
#module list

module purge
module load python-env/intelpython3.6-2018.3 gcc/5.4.0
module list


srun python preprocess.py

echo -e "\n ... \n training ended \n ... \n printing job stats .... \n"
used_slurm_resources.bash