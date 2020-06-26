#!/usr/bin/env bash

#SBATCH -J train
#SBATCH --mem-per-cpu 150000
#SBATCH --account=project_2001284
#SBATCH --gres=gpu:v100:1
#SBATCH -p gpu
#SBATCH -t 01:00:00

id -a

module purge
module load pytorch
module list

python -u inference.py -model_path ./checkpoint/FL_Pham34-090719-gn-aug/model_e121_73.30.pt -save_dir ./inference/FL_Pham34-090719-gn-aug-e121-2 -gpu 0

# python -u inference.py -model_path ./checkpoint/FL_Pham310-270319-test/model_1.pt \
#                        -save_dir ./inference/FL_Pham310-270319-test \
#                        -gpu 0

echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash
