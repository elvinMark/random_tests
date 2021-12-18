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
arch=WRN
optim=sam
for bs in 64 128 256 512 1024 2048 4096 8192 16384
do
    for lr in 0.1 0.2 0.4 0.8 1.6
    do
	for nbs in 0.0 0.01 0.02 0.04 0.08 0.16 0.32
	do
	    model_path="../SAM_tests/trained_models/WRN-28-10_with_original_SAM/model_epochs=165_lr=${lr}_momentum=0.9_weight_decay=0.0005_nbs=${nbs}_global_batch=${bs}_local_batch=$((bs/8)).ckpt"
	    # echo $model_path
	    mpirun -np 1 \
		   python3 calculate_hessian.py \
		   --dataset $ds \
		   --arch $arch \
		   --batch-size $bs \
		   --model_path $model_path \
		   --optim $optim \
		   --nbs $nbs \
		   --lr $lr 
	done
    done
done


# model_path="../SAM_tests/trained_models/WRN-28-10_with_original_SAM/model_epochs=165_lr=0.2_momentum=0.9_weight_decay=0.0005_nbs=0.08_global_batch=256_local_batch=32.ckpt"

# mpirun -np 1 \
#        python3 calculate_hessian.py \
#        --dataset $ds \
#        --arch $arch \
#        --batch-size $bs \
#        --model_path $model_path
