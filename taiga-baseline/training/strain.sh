#! /usr/bin/env bash

#SBATCH -J train
#SBATCH --account=project_2001284
#SBATCH --gres=gpu:v100:1
#SBATCH -p gpu
#SBATCH -t 72:00:00
#SBATCH --mem-per-cpu 150000

TAIGA_DIR=/scratch/project_2001284/TAIGA
DATA_DIR=../data/TAIGA

module -q purge
module load pytorch
module list

. ../venv/bin/activate

python3 -u train.py \
	    -hyper_data_path ${DATA_DIR}/hyperspectral_src.pt \
	    -src_norm_multiplier ${DATA_DIR}/image_norm_l2norm_along_channel.pt \
	    -tgt_path ${DATA_DIR}/hyperspectral_tgt_normalized.pt \
	    -metadata ${DATA_DIR}/metadata.pt \
            -hyper_data_header ${TAIGA_DIR}/20170615_reflectance_mosaic_128b.hdr \
	    -data_split_path ${DATA_DIR}/data-split \
	    -save_dir test_model \
	    -report_frequency 100 \
	    -model PhamModel3layers4 \
	    -input_normalize_method minmax_scaling \
	    -loss_balancing equal_weights \
	    -class_balancing focal_loss \
	    -augmentation flip \
	    -lr 0.0001 \
	    -gpu 0 \
	    -epoch 150 \
	    -patch_size 45 \
	    -patch_stride 13 \
	    -batch_size 32 \
	    -use_visdom \
	    -visdom_server http://puhti-login2.csc.fi \
	    -train_from ../checkpoint/test_model/model_e136_0.04546.pt

if [[ -v SLURM_JOB_ID ]]; then
    echo -e "\n ... printing job stats .... \n"
    seff $SLURM_JOB_ID
fi

