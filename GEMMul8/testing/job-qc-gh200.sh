#!/bin/bash
#SBATCH --job-name=test1             # 
#SBATCH --partition=qc-gh200         # 
#SBATCH -t 04:00:00                  # 

hostname  # 

make clean all
make test_f test_d MODE="all"
exit
