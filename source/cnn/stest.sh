#!/usr/bin/env bash

#SBATCH -J test
#SBATCH --mem-per-cpu 150000
#SBATCH --gres=gpu:v100:1
#SBATCH -p gpu
#SBATCH -t 1:00:00
#SBATCH --account=project_2001284

id -a

module purge
module load pytorch

module list

python -u test.py

echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash

