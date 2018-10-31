#!/usr/bin/env bash

#SBATCH -J train
#SBATCH --mem-per-cpu 10000
#SBATCH --gres=gpu:k80:1
#SBATCH -p gputest
#SBATCH -t 0:15:00

id -a

module purge
module load intelconda/python3.6-2018.3
module list

#env

python -u train.py  -src_path ../../data/hyperspectral_src_sm_l2norm2.pt \
                    -tgt_path ../../data/hyperspectral_tgt_sm.pt \
                    -metadata ../../data/metadata.pt \
                    -patch_size 35 \
                    -patch_step 2 \
                    -lr 0.001 \
                    -batch_size 64 \
                    -epoch 50 \
                    -model ChenModel \
                    -save_dir Chen-311018-test1 \
                    -report_frequency 5 \
                    -gpu 0

echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash
