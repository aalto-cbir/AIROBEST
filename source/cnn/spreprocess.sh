#!/usr/bin/env bash

#SBATCH -J preprocess
#SBATCH --mem-per-cpu 250000
##SBATCH --gres=gpu:k80:1
##SBATCH -p gputest
##SBATCH -t 0:15:00

#SBATCH --gres=gpu:k80:1
#SBATCH -p gpu
#SBATCH -t 02:00:00

#module purge
#module load intelconda/python3.6-2018.3
#module list

module purge
module load python-env/intelpython3.6-2018.3 gcc/5.4.0
module list


srun python -u preprocess.py    -src_file_name hyperspectral_src_full \
                                -tgt_file_name hyperspectral_tgt_full \
                                -normalize_method l2norm_along_channel \
                                -hyper_data_path /proj/deepsat/hyperspectral/20170615_reflectance_mosaic_128b.hdr \
                                -forest_data_path /proj/deepsat/hyperspectral/forestdata.hdr

echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash

