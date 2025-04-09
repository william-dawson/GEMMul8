#!/bin/bash
#SBATCH --job-name=test1             # 
#SBATCH --partition=qc-a100          # 
#SBATCH --gpus=1                     # 
#SBATCH -t 08:00:00                  # 

hostname  # 

make clean all
make test_f test_d MODE="all"
exit
