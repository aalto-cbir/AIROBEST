#!/usr/bin/env bash

#SBATCH -J train
#SBATCH --mem-per-cpu 150000
#SBATCH --gres=gpu:p100:2
#SBATCH -p gpu
#SBATCH -t 5:00:00
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
                    -input_normalize_method minmax_scaling \
                    -report_frequency 100 \
                    -visdom_server http://taito-gpu.csc.fi \
                    -use_visdom \
                    -lr 0.0001 \
                    -data_split_path ${DATA_DIR}/splits-orig \
                    -gpu 0 \
                    -epoch 200 \
                    -model PhamModel3layers10 \
                    -patch_size 27 \
                    -patch_stride 27 \
                    -batch_size 64 \
                    -save_dir FL_Pham310-110619-ew-no-aug \
                    -loss_balancing equal_weights \
                    -class_balancing focal_loss

#                    -ignored_cls_tasks 0 1
#                    -ignored_reg_tasks 0 1 2 3 4 5 6 7 9 10 11 12 13 14
echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash
