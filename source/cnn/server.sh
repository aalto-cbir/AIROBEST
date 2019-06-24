#!/usr/bin/env bash

module purge
module load python-env/intelpython3.6-2018.3 gcc/5.4.0
module list

python -m visdom.server -port 8097