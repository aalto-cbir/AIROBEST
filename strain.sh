#!/usr/bin/env bash

#SBATCH -J train
#SBATCH --mem-per-cpu 10000
##SBATCH --gres=gpu:k80:1
##SBATCH -p gputest
##SBATCH -t 0:15:00

#SBATCH --gres=gpu:p100:1
#SBATCH -p gpu
#SBATCH -t 4:00:00

id -a

module purge
module load intelconda/python3.6-2018.3
module list

#env

python -u train.py  -src_path ./data/hyperspectral_src_l2norm.pt \
                    -tgt_path ./data/hyperspectral_tgt.pt \
                    -patch_size 1 \
                    -patch_step 1 \
                    -lr 1e-4 \
                    -batch_size 14630 \
                    -epoch 22 \
                    -model ChenModel \
                    -save_dir Chen-model-1 \
                    -gpu 0

echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash
