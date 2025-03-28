#!/bin/bash
#SBATCH --job-name=flops_watt_test   # 
#SBATCH --partition=a100             # 
#SBATCH -t 04:00:00                  # 

hostname  # 

make clean all
make test_d MODE="flops_check watt_check"

