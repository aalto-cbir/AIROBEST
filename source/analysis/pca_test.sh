#! /bin/bash

#SBATCH --mem=16G
#SBATCH --time=30:00

hostname
unset DISPLAY

module purge 
module load python-env/3.5.3

echo $PYTHONPATH
echo $DISPLAY

./pca_test.py $*

