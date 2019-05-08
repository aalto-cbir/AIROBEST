#!/usr/bin/env bash

#SBATCH -J train
#SBATCH --mem-per-cpu 150000
#SBATCH --gres=gpu:p100:2
#SBATCH -p gpu
#SBATCH -t 4:00:00
##SBATCH --begin=02:00

id -a

module purge
module load python-env/3.6.3-ml
#module load intelconda/python3.6-2018.3
module list

#env
DATA_DIR=./data/mosaic-all-bands

python -u train.py  -hyper_data_path ${DATA_DIR}/hyperspectral_src.pt \
                    -src_norm_multiplier ${DATA_DIR}/image_norm_l2norm_along_channel.pt \
                    -tgt_path ${DATA_DIR}/hyperspectral_tgt_normalized.pt \
                    -metadata ${DATA_DIR}/metadata.pt \
                    -model PhamModel3layers4 \
                    -patch_size 27 \
                    -patch_stride 27 \
                    -lr 0.0001 \
                    -batch_size 64 \
                    -epoch 200 \
                    -input_normalize_method minmax_scaling \
                    -save_dir CRL_Pham34-080519-critic-balanced-acc-all-bands-l2reg-kappa8 \
                    -report_frequency 100 \
                    -loss_balancing equal_weights \
                    -visdom_server http://taito-gpu.csc.fi \
                    -use_visdom \
                    -class_balancing CRL \
                    -gpu 0

#                    -ignored_cls_tasks 0 1
#                    -ignored_reg_tasks 0 1 2 3 4 5 6 7 9 10 11 12 13 14
echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash
