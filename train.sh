#!/bin/bash
#YBATCH -r any_1
#SBATCH -N 1
#SBATCH -J tests
#SBATCH --output output/%j.out

. /etc/profile.d/modules.sh
module load openmpi/3.1.6 cuda/11.1 nccl/cuda-11.1/2.7.8

export MASTER_ADDR=$(ifconfig | grep inet | grep 192.168.205 | cut -d " " -f 10)
export MASTER_PORT=3535

export CIFAR10_PATH=../data/cifar-10-python/

export NGPUS=1

mpirun -np $NGPUS \
python3 train.py ./ \
       --dataset CIFAR10 \
       --model myresnet18 \
       --num-classes 10 \
       --input-size 3 32 32 \
       --batch-size 32 \
       --momentum 0.9 \
       --sched step \
       --lr 0.1 \
       --epochs 2 \
       --output ./trained_models/ \
       --experiment cifar10_resnet18_step_noiseless \
       --log-wandb \
       --project lr_noise_tests \
       --ngpus 1
