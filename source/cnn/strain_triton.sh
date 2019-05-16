#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 03:00:00
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
                    -model PhamModel3layers8 \
                    -patch_size 27 \
                    -patch_stride 27 \
                    -lr 0.0001 \
                    -batch_size 64 \
                    -epoch 200 \
                    -input_normalize_method minmax_scaling \
                    -save_dir FL_Pham38-150519-critic-balanced-acc-all-bands-l2reg-cont \
                    -report_frequency 100 \
                    -loss_balancing equal_weights \
                    -visdom_server http://login2.triton.aalto.fi \
                    -class_balancing focal_loss \
                    -gpu 0 \
                    -augmentation flip \
                    -train_from ./checkpoint/FL_Pham38-150519-critic-balanced-acc-all-bands-l2reg/model_100.pt
