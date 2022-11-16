#! /usr/bin/env bash

#SBATCH -J preprocess
#SBATCH --account=project_2001284
#SBATCH --gres=gpu:v100:1
#SBATCH -p gpu
#SBATCH -t 1:30:00
#SBATCH --mem-per-cpu 200000

TAIGA_DIR=/scratch/project_2001284/TAIGA
DATA_DIR=../data/TAIGA

module -q purge
module load pytorch
module list

. ../venv/bin/activate

python3 -u preprocess.py \
	    --data_dir $TAIGA_DIR \
	    --hyperspec 20170615_reflectance_mosaic_128b.hdr \
	    --forestdata forestdata_stands.hdr \
	    --save_dir $DATA_DIR \
	    --src_file_name hyperspectral_src \
	    --tgt_file_name hyperspectral_tgt_normalized \
	    --metadata_file_name metadata \
	    --input_normalize_method l2norm_along_channel \
	    --output_normalize_method clip \
	    --ignore_zero_labels

if [[ -v SLURM_JOB_ID ]]; then
   echo -e "\n ... printing job stats .... \n"
   seff $SLURM_JOB_ID
fi

