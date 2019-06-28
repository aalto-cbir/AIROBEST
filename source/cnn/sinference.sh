#!/usr/bin/env bash

#SBATCH -J train
#SBATCH --mem-per-cpu 50000
##SBATCH --gres=gpu:k80:2
##SBATCH -p gputest

#SBATCH --gres=gpu:p100:1
#SBATCH -p gpu
#SBATCH -t 00:15:00

id -a

module purge
module load python-env/3.6.3-ml
module list

python -u inference.py -model_path ./checkpoint/FL_Pham310-270319-test/model_1.pt \
                       -save_dir ./inference/FL_Pham310-270319-test \
                       -gpu 0

echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash
