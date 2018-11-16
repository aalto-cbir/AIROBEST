#!/usr/bin/env bash

#SBATCH -J train
#SBATCH --mem-per-cpu 250000
#SBATCH --gres=gpu:k80:1
#SBATCH -p gpu
#SBATCH -t 2:00:00

id -a

module purge
module load intelconda/python3.6-2018.3
module list

#env

python -u train.py  -src_path data/hyperspectral_src_full_l2norm_along_channel.pt \
                    -tgt_path data/hyperspectral_tgt_full.pt \
                    -metadata data/metadata.pt \
                    -patch_size 35 \
                    -patch_step 5 \
                    -lr 0.001 \
                    -batch_size 64 \
                    -epoch 50 \
                    -model ChenModel \
                    -save_dir Chen-model-141118 \
                    -report_frequency 50 \
                    -visdom_server http://taito-gpu.csc.fi \
                    -gpu 0

echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash
