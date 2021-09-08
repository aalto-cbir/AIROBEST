#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 05:00:00
#SBATCH -J gpu_job
#SBATCH --gres=gpu:v100:2
#SBATCH --mem-per-cpu=150G

source activate pytorch

#env

DATA_DIR=./data/mosaic-full
srun python -u train.py  -hyper_data_path ${DATA_DIR}/hyperspectral_src.pt \
                    -src_norm_multiplier ${DATA_DIR}/image_norm_l2norm_along_channel.pt \
                    -tgt_path ${DATA_DIR}/hyperspectral_tgt_normalized.pt \
                    -metadata ${DATA_DIR}/metadata.pt \
                    -hyper_data_header /scratch/work/phama1/deepsat/hypdata/20170615_reflectance_mosaic_128b.hdr \
                    -input_normalize_method minmax_scaling \
                    -epoch 200 \
                    -lr 0.0001 \
                    -data_split_path ${DATA_DIR}/splits \
                    -patch_size 27 \
                    -patch_stride 27 \
                    -batch_size 64 \
                    -model PhamModel3layers4 \
                    -save_dir FL_Pham4-200519-uncertainty \
                    -report_frequency 100 \
                    -loss_balancing uncertainty \
                    -visdom_server http://login2.triton.aalto.fi \
                    -class_balancing focal_loss \
                    -gpu 0
