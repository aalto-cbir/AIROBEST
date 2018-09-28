#!/usr/bin/env bash

#SBATCH -J preprocess
#SBATCH --mem-per-cpu 20000
#SBATCH --gres=gpu:p100:1
#SBATCH -p gpu
#SBATCH -t 0:15:00

module purge
module load intelconda/python3.6-2018.3
module list

srun python preprocess.py