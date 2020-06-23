#!/usr/bin/env bash

#SBATCH -J train
#SBATCH --mem-per-cpu 15000
#SBATCH --gres=gpu:k80:2
#SBATCH -p gputest
#SBATCH -t 00:15:00

##SBATCH --gres=gpu:p100:1
##SBATCH -p gpu
##SBATCH -t 0:15:00

id -a

module purge
module load python-env/3.6.3-ml
module list

#env
DATA_DIR=./data/subsetA-full-bands/

python -u train.py  -hyper_data_path ${DATA_DIR}/hyperspectral_src.pt \
                    -src_norm_multiplier ${DATA_DIR}/image_norm_l2norm_along_channel.pt \
                    -tgt_path ${DATA_DIR}/hyperspectral_tgt_normalized.pt \
                    -metadata ${DATA_DIR}/metadata.pt \
                    -patch_size 29 \
                    -patch_stride 29 \
                    -data_split_path ${DATA_DIR}/splits-orig \
                    -lr 0.0001 \
                    -batch_size 64 \
                    -epoch 50 \
                    -model PhamModel3layers10 \
                    -input_normalize_method minmax_scaling \
                    -save_dir FL_Pham310-270319-test \
                    -report_frequency 5 \
                    -loss_balancing uncertainty \
                    -visdom_server http://taito-gpu.csc.fi \
                    -use_visdom \
                    -class_balancing focal_loss \
                    -gpu 0
#                    -ignored_cls_tasks 1 \
#                    -ignored_reg_tasks 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14

echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash
