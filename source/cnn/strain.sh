#!/usr/bin/env bash

#SBATCH -J train5
#SBATCH --mem-per-cpu 150000
#SBATCH --gres=gpu:v100:1
#SBATCH -p gpu
#SBATCH -t 72:00:00
#SBATCH --account=project_2001284

id -a
module purge
module load pytorch/1.4
module list

#env
DATA_DIR=./data/TAIGA

python -u train.py -hyper_data_path ${DATA_DIR}/hyperspectral_src.pt -src_norm_multiplier ${DATA_DIR}/image_norm_l2norm_along_channel.pt -tgt_path ${DATA_DIR}/hyperspectral_tgt_normalized.pt -metadata ${DATA_DIR}/metadata.pt -hyper_data_header /scratch/project_2001284/hyperspectral/20170615_reflectance_mosaic_128b.hdr -input_normalize_method minmax_scaling -report_frequency 100 -lr 0.0001 -data_split_path ${DATA_DIR}/eelis-split13-2 -gpu 0 -epoch 150 -model PhamModel3layers4 -patch_size 45 -patch_stride 45 -batch_size 32 -save_dir S2-45_model_only_fertility_class -loss_balancing equal_weights -class_balancing focal_loss -augmentation flip -keep_best 10 -visdom_server http://puhti-login2.csc.fi/ -ignored_cls_tasks 4 5 6 7 8 -ignored_reg_tasks 0 1 2 3 4 5 6 7 8 9 10 11 12
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
used_slurm_resources.bash



# python -u train.py  -hyper_data_path ./data/subsetA-full-bands/hyperspectral_src.pt -src_norm_multiplier ./data/subsetA-full-bands/image_norm_l2norm_along_channel.pt -tgt_path ./data/subsetA-full-bands/hyperspectral_tgt_normalized.pt -metadata ./data/subsetA-full-bands/metadata.pt -input_normalize_method minmax_scaling -report_frequency 100 -visdom_server http://localhost -use_visdom -lr 0.0001 -data_split_path ./data/subsetA-full-bands/splits-orig -gpu 0 -epoch 200 -model PhamModel3layers10 -patch_size 91 -patch_stride 91 -batch_size 64 -save_dir FL_Pham310-110619-ew-no-aug -loss_balancing equal_weights -class_balancing focal_loss -train_from ./checkpoint/FL_Pham310-110619-ew-no-aug/model_1.pt
