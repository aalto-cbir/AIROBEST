#! /usr/bin/env bash

#SBATCH -J train5
#SBATCH --mem-per-cpu 150000
#SBATCH --gres=gpu:v100:1
#SBATCH -p gpu
#SBATCH -t 72:00:00
#SBATCH --account=project_2001284

id -a
module purge
module load pytorch/1.10
module list
DATA_DIR=../data/TAIGA
. ../venv/bin/activate

python3 -u train.py \
	    -hyper_data_path ${DATA_DIR}/hyperspectral_src.pt \
	    -src_norm_multiplier ${DATA_DIR}/image_norm_l2norm_along_channel.pt \
	    -tgt_path ${DATA_DIR}/hyperspectral_tgt_normalized.pt \
	    -metadata ${DATA_DIR}/metadata.pt \
        -hyper_data_header /scratch/project_2001284/TAIGA/20170615_reflectance_mosaic_128b.hdr \
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
    
# -keep_best 10 -visdom_server http://puhti-login2.csc.fi/ -ignored_cls_tasks 4 5 6 7 8 -ignored_reg_tasks 0 1 2 3 4 5 6 7 8 9 10 11 12
# -ignored_cls_tasks 4 5 6 7 8 -ignored_reg_tasks 0 1 2 3 4 5 6 7 8 9 10 11 12
# -ignored_cls_tasks 0 1 2 3 4 5 6 7 8
# -ignored_reg_tasks 4 5 6 10 11 12

# 
# -use_visdom 

#-train_from ./checkpoint/FL_Pham34-18012021-gn-aug-eelis-patchsize91-set1-flip/model_e96_0.09109.pt
# -ignored_cls_tasks 0 1 2 9 -ignored_reg_tasks 0 1 2 3 4 5 6 7 8 10 11 12 13 14

#                    -ignored_cls_tasks 0 1
#                    -ignored_reg_tasks 0 1 2 3 4 5 6 7 9 10 11 12 13 14

echo -e "\n ... printing job stats .... \n"

seff $SLURM_JOB_ID
