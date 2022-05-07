#!/usr/bin/env bash

#SBATCH -J preprocess
#SBATCH --account=project_2001284
#SBATCH --gres=gpu:v100:1

#SBATCH -p gpu
#SBATCH -t 1:30:00
#SBATCH --mem-per-cpu 200000

##SBATCH -p gputest
##SBATCH -t 0:15:00
##SBATCH --mem-per-cpu 5000

module purge
#module load intelconda/python3.6-2018.3
#module load python-env/intelpython3.6-2018.3 gcc/5.4.0
module load pytorch
module list


SAVE_DIR=new_TAIGA
HYP_IMAGE=20170615_reflectance_mosaic_128b.hdr

# . venv/bin/activate

python -u preprocess.py --data_dir /scratch/project_2001284/hyperspectral --save_dir $SAVE_DIR --hyperspec $HYP_IMAGE --forestdata forestdata_stands.hdr --src_file_name hyperspectral_src --tgt_file_name hyperspectral_tgt_normalized --metadata_file_name metadata --normalize_method l2norm_along_channel --ignore_zero_labels --remove_bad_data --label_normalize_method clip
            #--hyperspec_bands 0:110

echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash







############################################


# #! /bin/bash

# #SBATCH -J preprocess
# #SBATCH --account=project_2001284
# #SBATCH --gres=gpu:v100:1
# #SBATCH -p gpu
# #SBATCH -t 1:30:00
# #SBATCH --mem-per-cpu 20G

# ##SBATCH -p gputest
# ##SBATCH -t 0:15:00
# ##SBATCH --mem-per-cpu 5000

# #module purge
# #module load intelconda/python3.6-2018.3
# #module load python-env/intelpython3.6-2018.3 gcc/5.4.0
# #module load pytorch/1.3.1
# #module list

# SAVE_DIR=mosaic6
# HYP_IMAGE=20170615_reflectance_mosaic_128b.hdr

# . venv/bin/activate

# python -u preprocess.py \
#             --data_dir /scratch/project_2001284/hyperspectral \
#             --save_dir $SAVE_DIR \
#             --hyperspec $HYP_IMAGE \
#             --forestdata forestdata_phu.hdr \
#             --src_file_name hyperspectral_src \
#             --ignored_bands 3 8 \
#             --tgt_file_name hyperspectral_tgt_normalized \
#             --metadata_file_name metadata \
#             --normalize_method l2norm_along_channel \
#             --ignore_zero_labels \
#             --remove_bad_data \
#             --label_normalize_method clip

# echo -e "\n ... printing job stats .... \n"
# used_slurm_resources.bash
