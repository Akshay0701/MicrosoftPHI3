#!/bin/bash

eval "$(/home/$(whoami)/miniconda3/bin/conda shell.bash hook)"
conda activate phi3_env

python3 cephalo_fine_tune.py