#!/bin/bash
#YBATCH -r am_2
#SBATCH -N 1
#SBATCH -J sam_test
#SBATCH --output output/%j.out

# ======== Modules ========
. /etc/profile.d/modules.sh
module load cuda/11.1 cudnn/cuda-11.1/8.0 nccl/cuda-11.1/2.7.8 openmpi/3.1.6

export MASTER_ADDR=$(/usr/sbin/ip a show | grep inet | grep 192.168.205 | head -1 | cut -d " " -f 6 | cut -d "/" -f 1)
export MASTER_PORT=3535

export NGPUS=2
export NUM_PROC=2

# Single Runs Imagenet

# Batch Size
bs=32768
# Learning Rate
lr=0.8
# Neighbour radius
# nbs=0.16
# optimizer
optim=sam
# dataset
dataset=CIFAR10
# architecture
arch=ConvMixer

for nbs in 0.0 0.16
do
    mpirun -npernode $NUM_PROC -np $NGPUS \
	   python main.py \
	   --dataset $dataset \
	   --arch $arch \
	   --dist-url $MASTER_ADDR \
	   --epochs 140 \
	   --batch-size $bs \
	   --nbs $nbs \
	   --lr $lr \
	   --sched multistep \
	   --milestones 60 90 120 \
	   --lower_lr 0.0005 \
	   --warmup 10 \
	   --experiment ${dataset}_bs_${bs}_ngpus_${NGPUS}_lr_${lr}_nbs_${nbs}_optim_${optim}_arch_${arch} \
	   --project exp_test
done


# # Single Runs

# # Batch Size
# bs=128
# # Learning Rate
# lr=0.01
# # Neighbour radius
# nbs=0.0

# mpirun -npernode $NUM_PROC -np $NGPUS \
#        python main.py \
#        --dataset Imagenet \
#        --arch Resnet \
#        --dist-url $MASTER_ADDR \
#        --epochs 140 \
#        --batch-size $bs \
#        --nbs $nbs \
#        --lr $lr \
#        --sched multistep \
#        --milestones 60 90 120 \
#        --lower_lr 0.05 \
#        --warmup 25 \
#        --experiment tiny_bs_${bs}_ngpus_${NGPUS}_lr_${lr}_nbs_${nbs} \
#        --project exp_test


# Loops runs

# # Batch Size
# bs=256
# # Learning Rate
# # lr=1.2
# # Neighbour radius
# # nbs=0.04
# # r

# for lr in 0.1 0.2 0.4 0.8 1.6
# do
#     for nbs in 0.0 0.01 0.02 0.04 0.08 0.16 0.32
#     do
	
# 	mpirun -npernode $NUM_PROC -np $NGPUS \
# 	       python main.py \
# 	       --dist-url $MASTER_ADDR \
# 	       --epochs 140 \
# 	       --batch-size $bs \
# 	       --nbs $nbs \
# 	       --lr $lr \
# 	       --sched multistep \
# 	       --milestones 60 90 120 \
# 	       --lower_lr 0.05 \
# 	       --warmup 25 \
# 	       --experiment bs_${bs}_ngpus_${NGPUS}_lr_${lr}_nbs_${nbs} \
# 	       --project exp_9
#     done
# done
