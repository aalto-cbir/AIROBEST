#!/usr/bin/env bash

#SBATCH -J inference5
#SBATCH --mem-per-cpu 150000
#SBATCH --account=project_2001284
#SBATCH --gres=gpu:v100:1
#SBATCH -p gpu
#SBATCH -t 72:00:00

id -a

module purge
module load pytorch/1.4
module list

python -W ignore -u inference.py -model_path ./checkpoint/S2-45_model_only_cont/model_e125_0.05967.pt -save_dir ./inference/S2-45_model_only_cont -gpu 0

# python -u inference.py -model_path ./checkpoint/FL_Pham310-270319-test/model_1.pt \
#                        -save_dir ./inference/FL_Pham310-270319-test \
#                        -gpu 0

echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash
