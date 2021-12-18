#!/bin/bash
#YBATCH -r any_1
#SBATCH -N 1
#SBATCH -J test
#SBATCH --output output/%j.out

. /etc/profile.d/modules.sh
module load cuda/11.1 cudnn/cuda-11.1/8.0 nccl/cuda-11.1/2.7.8 openmpi/3.1.6                                                               

export MASTER_ADDR=$(/usr/sbin/ip a show | grep inet | grep 192.168.205 | head -1 | cut -d " " -f 6 | cut -d "/" -f 1)
export MASTER_PORT=3555

ds=CIFAR10
optim=sam
arch=WRN
bs=128

model_path="../SAM_tests/trained_models/WRN-28-10_with_original_SAM/model_epochs=165_lr=0.1_momentum=0.9_weight_decay=0.0005_nbs=0.04_global_batch=${bs}_local_batch=$((bs/8)).ckpt"
mpirun -np 1 \
       python3 mytest.py \
       --dataset $ds \
       --arch $arch \
       --optim $optim \
       --batch-size $bs \
       --model_path $model_path 
