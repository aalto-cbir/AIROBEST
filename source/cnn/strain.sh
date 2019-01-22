#!/usr/bin/env bash

#SBATCH -J train
#SBATCH --mem-per-cpu 100000
#SBATCH --gres=gpu:p100:1
#SBATCH -p gpu
#SBATCH -t 5:00:00
##SBATCH --begin=07:00

id -a

module purge
module load intelconda/python3.6-2018.3
module list

#env
DATA_DIR=./data/mosaic

python -u train.py  -hyper_data_path ${DATA_DIR}/hyperspectral_src.pt \
                    -src_norm_multiplier ${DATA_DIR}/image_norm_l2norm_along_channel.pt \
                    -tgt_path ${DATA_DIR}/hyperspectral_tgt_normalized.pt \
                    -metadata ${DATA_DIR}/metadata.pt \
                    -patch_size 27 \
                    -patch_stride 2 \
                    -lr 0.0001 \
                    -batch_size 64 \
                    -epoch 150 \
                    -model PhamModel \
                    -input_normalize_method minmax_scaling \
                    -save_dir Pham-210119-f1-cb-gn \
                    -report_frequency 150 \
                    -loss_balancing grad_norm \
                    -visdom_server http://taito-gpu.csc.fi \
                    -use_visdom \
                    -class_balancing \
                    -gpu 0

echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash
