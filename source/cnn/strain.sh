#!/usr/bin/env bash

#SBATCH -J train
#SBATCH --mem-per-cpu 10000
#SBATCH --gres=gpu:p100:1
#SBATCH -p gpu
#SBATCH -t 5:00:00

id -a

module purge
module load intelconda/python3.6-2018.3
module list

#env

python -u train.py  -hyper_data_path /proj/deepsat/hyperspectral/20170615_reflectance_mosaic_128b.hdr \
                    -src_norm_multiplier ./data/hyperspectral_src_l2norm_along_channel.pt \
                    -tgt_path ./data/hyperspectral_tgt_full.pt \
                    -metadata ./data/metadata_full.pt \
                    -patch_size 27 \
                    -patch_stride 2 \
                    -lr 0.0001 \
                    -batch_size 64 \
                    -epoch 50 \
                    -model ChenModel \
                    -save_dir Chen-model-181218 \
                    -report_frequency 50 \
                    -visdom_server http://taito-gpu.csc.fi \
                    -gpu 0

echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash
