#!/usr/bin/env bash

#SBATCH -J train
#SBATCH --mem-per-cpu 50000
#SBATCH --gres=gpu:p100:1
#SBATCH -p gpu
#SBATCH -t 5:00:00

id -a

module purge
module load intelconda/python3.6-2018.3
module list

#env

python -u train.py -src_path ./data/hyperspectral_src.pt \
                     -tgt_path ./data/hyperspectral_tgt.pt \
                     -gpu 0 \
                     -epoch 20

echo -e "\n ... \n training ended \n ... \n printing job stats .... \n"
used_slurm_resources.bash
