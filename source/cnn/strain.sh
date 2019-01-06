#!/usr/bin/env bash

#SBATCH -J train
#SBATCH --mem-per-cpu 100000
#SBATCH --gres=gpu:p100:1
#SBATCH -p gpu
#SBATCH -t 10:00:00

id -a

module purge
module load intelconda/python3.6-2018.3
module list

#env

python -u train.py  -hyper_data_path ./data/hyper_image.pt \
                    -src_norm_multiplier ./data/hyperspectral_src_l2norm_along_channel.pt \
                    -tgt_path ./data/hyperspectral_tgt_full_normalized.pt \
                    -metadata ./data/metadata_full.pt \
                    -patch_size 27 \
                    -patch_stride 2 \
                    -lr 0.0001 \
                    -batch_size 64 \
                    -epoch 50 \
                    -model PhamModel \
                    -save_dir Pham-050119-2 \
                    -report_frequency 150 \
                    -loss_balancing equal_weights \
                    -visdom_server http://taito-gpu.csc.fi \
                    -gpu 0

echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash
