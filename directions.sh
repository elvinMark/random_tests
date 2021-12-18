#!/bin/bash
#YBATCH -r any_1
#SBATCH -N 1
#SBATCH -J test
#SBATCH --output output/%j.out

. /etc/profile.d/modules.sh
module load cuda/11.1 cudnn/cuda-11.1/8.0 nccl/cuda-11.1/2.7.8 openmpi/3.1.6                                                               

export MASTER_ADDR=$(/usr/sbin/ip a show | grep inet | grep 192.168.205 | head -1 | cut -d " " -f 6 | cut -d "/" -f 1)
export MASTER_PORT=3555


models_list=models_list.txt
mpirun -np 1 \
       python3 directions.py \
       --models-list $models_list
