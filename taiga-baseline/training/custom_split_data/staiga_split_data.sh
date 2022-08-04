#! /usr/bin/env bash

#SBATCH -J create_full_dataset2
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

python3 -u taiga_split_data.py

echo -e "\n ... printing job stats .... \n"

seff $SLURM_JOB_ID
