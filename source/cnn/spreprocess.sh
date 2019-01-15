#!/usr/bin/env bash

#SBATCH -J preprocess
##SBATCH --mem-per-cpu 5000
##SBATCH -p gputest
##SBATCH -t 0:15:00

#SBATCH --mem-per-cpu 200000
#SBATCH -p gpu
#SBATCH -t 1:30:00

#module purge
#module load intelconda/python3.6-2018.3
#module list

module purge
module load python-env/intelpython3.6-2018.3 gcc/5.4.0
module list

#SAVE_DIR=subsetA
#HYP_IMAGE=subset_A_20170615_reflectance.hdr
SAVE_DIR=mosaic
HYP_IMAGE=20170615_reflectance_mosaic_128b.hdr

srun python -u preprocess.py    -save_dir ${SAVE_DIR} \
                                -src_file_name hyperspectral_src \
                                -categorical_bands 0 1 2 9 \
                                -tgt_file_name hyperspectral_tgt_normalized \
                                -metadata_file_name metadata \
                                -normalize_method l2norm_along_channel \
                                -hyper_data_path /proj/deepsat/hyperspectral/${HYP_IMAGE} \
                                -forest_data_path /proj/deepsat/hyperspectral/forestdata.hdr \
                                -ignore_zero_labels True \
                                -label_normalize_method clip

echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash
