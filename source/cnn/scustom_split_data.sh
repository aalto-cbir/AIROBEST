#!/usr/bin/env bash

#SBATCH -J create_full_dataset
#SBATCH --mem-per-cpu 150000
#SBATCH --account=project_2001284
#SBATCH --gres=gpu:v100:1
#SBATCH -p gpu
#SBATCH -t 12:00:00

id -a
module purge
module load pytorch
module list
python -u custom_split_data.py


echo -e "\n ... printing job stats .... \n"

used_slurm_resources.bash
