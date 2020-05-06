#!/usr/bin/env bash

module purge
module load pytorch
module list

python -m visdom.server -port 8097

