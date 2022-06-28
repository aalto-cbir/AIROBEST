#! /usr/bin/env bash

#SBATCH -J create_full_dataset
#SBATCH --mem-per-cpu 150000
#SBATCH --account=project_2001284
#SBATCH --gres=gpu:v100:1
#SBATCH -p gpu
#SBATCH -t 12:00:00

id -a
module purge
module load pytorch/1.10
module list

. ../../venv/bin/activate

python -u custom_split_data.py -patch_size 45 -new_patch_size 13 -data_path ../../data/TAIGA

echo -e "\n ... printing job stats .... \n"

seff $SLURM_JOB_ID
