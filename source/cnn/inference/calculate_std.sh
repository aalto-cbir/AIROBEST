#!/usr/bin/env bash

#SBATCH -J calculate_std
#SBATCH --mem-per-cpu 150000
#SBATCH --gres=gpu:v100:1
#SBATCH -p gpu
#SBATCH -t 4:00:00
#SBATCH --account=project_2001284


id -a

module purge
module load pytorch
module list

python -u calculate_std.py 

echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash
