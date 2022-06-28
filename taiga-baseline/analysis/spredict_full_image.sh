#! /usr/bin/env bash

#SBATCH -J predict3
#SBATCH --mem-per-cpu 150000
#SBATCH --account=project_2001284
#SBATCH --gres=gpu:v100:1
#SBATCH -p gpu
#SBATCH -t 72:00:00

id -a
module purge
module load pytorch/1.10
module list

python -u predict_full_image.py \
       -model_path ./checkpoint/FL_Pham34-12012021-gn-aug-eelis-patchsize45/model_e132_95.81.pt \
       -image_set ./data/TAIGA/complete_image3.npy \
       -save_dir ./inference/FL_Pham34-12012021-gn-aug-eelis-patchsize45 \
       -file_name complete_image3.pt \
       -gpu 0

# python -u inference.py -model_path ./checkpoint/FL_Pham310-270319-test/model_1.pt \
#                        -save_dir ./inference/FL_Pham310-270319-test \
#                        -gpu 0

echo -e "\n ... printing job stats .... \n"
used_slurm_resources.bash
