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
module load pytorch/1.3.1
module list

#SAVE_DIR=subsetA4
#HYP_IMAGE=subset_A_20170615_reflectance.hdr
SAVE_DIR=mosaic6
HYP_IMAGE=20170615_reflectance_mosaic_128b.hdr

PROJ=/scratch/project_2001284

srun python -u preprocess.py    -save_dir ${SAVE_DIR} \
                                -src_file_name hyperspectral_src \
                                -categorical_bands 0 1 2 9 \
                                -ignored_bands 3 8 \
                                -tgt_file_name hyperspectral_tgt_normalized \
                                -metadata_file_name metadata \
                                -normalize_method l2norm_along_channel \
                                -hyper_data_path $PROJ/deepsat/hyperspectral/${HYP_IMAGE} \
                                -forest_data_path $PROJ/deepsat/hyperspectral/forestdata.hdr \
                                -ignore_zero_labels \
                                -remove_bad_data \
                                -label_normalize_method clip

echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash
