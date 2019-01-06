#!/usr/bin/env bash

#SBATCH -J train
#SBATCH --mem-per-cpu 100000
#SBATCH --gres=gpu:k80:1
#SBATCH -p gputest
#SBATCH -t 0:15:00

##SBATCH --gres=gpu:p100:1
##SBATCH -p gpu
##SBATCH -t 3:00:00

id -a

module purge
module load intelconda/python3.6-2018.3
module list

#env
DATA_DIR=./data/subsetA/

python -u train.py  -hyper_data_path ${DATA_DIR}/hyperspectral_src.pt \
                    -src_norm_multiplier ${DATA_DIR}/image_norm_l2norm_along_channel.pt \
                    -tgt_path ${DATA_DIR}/hyperspectral_tgt_normalized.pt \
                    -metadata ${DATA_DIR}/metadata.pt \
                    -patch_size 27 \
                    -patch_stride 2 \
                    -lr 0.0001 \
                    -batch_size 64 \
                    -epoch 50 \
                    -model PhamModel \
                    -save_dir Pham-060119-test1 \
                    -report_frequency 5 \
                    -loss_balancing equal_weights \
                    -visdom_server http://taito-gpu.csc.fi \
                    -gpu 0

echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash
