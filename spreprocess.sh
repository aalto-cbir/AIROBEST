#!/usr/bin/env bash

#SBATCH -J preprocess
#SBATCH --mem-per-cpu 10000
#SBATCH --gres=gpu:k80:1
#SBATCH -p gputest
#SBATCH -t 0:15:00

#module purge
#module load intelconda/python3.6-2018.3
#module list

module purge
module load python-env/intelpython3.6-2018.3 gcc/5.4.0
module list


srun python -u preprocess.py    -src_file_name hyperspectral_src_l2norm \
                                -tgt_file_name hyperspectral_tgt

echo -e "\n ... \n training ended \n ... \n printing job stats .... \n"
used_slurm_resources.bash