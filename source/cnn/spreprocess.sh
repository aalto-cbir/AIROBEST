#!/usr/bin/env bash

#SBATCH -J preprocess
#SBATCH --mem-per-cpu 50000
#SBATCH --gres=gpu:k80:1
#SBATCH -p gputest
#SBATCH -t 0:15:00

##SBATCH --gres=gpu:k80:1
##SBATCH -p gpu
##SBATCH -t 02:00:00

#module purge
#module load intelconda/python3.6-2018.3
#module list

module purge
module load python-env/intelpython3.6-2018.3 gcc/5.4.0
module list


srun python -u preprocess.py    -src_file_name hyperspectral_src_subA_sm \
                                -tgt_file_name hyperspectral_tgt_subA_sm \
                                -metadata_file_name metadata_subA_sm \
                                -normalize_method l2norm_along_channel \
                                -hyper_data_path /proj/deepsat/hyperspectral/subset_A_20170615_reflectance.hdr \
                                -forest_data_path /proj/deepsat/hyperspectral/forestdata.hdr \
                                -sharding_size_along_row 5 \
                                -sharding_size_along_col 5

echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash

