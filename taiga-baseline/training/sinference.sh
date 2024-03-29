#! /usr/bin/env bash

#SBATCH -J inference5
#SBATCH --mem-per-cpu 150000
#SBATCH --account=project_2001284
#SBATCH --gres=gpu:v100:1
#SBATCH -p gpu
#SBATCH -t 72:00:00

id -a
module purge
module load pytorch/1.10
module list
. ../venv/bin/activate
model=../checkpoint/test_model/model_e150_0.04576.pt

python3 -W ignore \
	    -u inference.py \
	    -model_path $model \
	    -save_dir ../inference/S2-45_model \
	    -gpu 0

echo -e "\n ... printing job stats .... \n"

seff $SLURM_JOB_ID
