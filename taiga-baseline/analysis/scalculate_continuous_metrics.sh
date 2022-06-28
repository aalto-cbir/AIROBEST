#! /usr/bin/env bash

#SBATCH -J calculate_continuous_metrics
#SBATCH --mem-per-cpu 150000
#SBATCH --account=project_2001284
#SBATCH -t 3:00:00

. ../venv/bin/activate
./calculate_continuous_metrics.py --pred ../inference/S2-45_model/test_pred.pt \
                                  --test_set ../data/TAIGA/data-split/test_set.npy \
                                  --stand_ids ../data/TAIGA/stand_ids.npy

