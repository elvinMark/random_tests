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
pca_path="directions/"
arch=WRN
bs=128

for model_path in $(cat list_models.txt)
do
    # model_path="../SAM_tests/trained_models/WRN-28-10_with_original_SAM/model_epochs=165_lr=${lr}_momentum=0.9_weight_decay=0.0005_nbs=0.0_global_batch=${bs}_local_batch=$((bs/8)).ckpt"
    echo $model_path
    mpirun -np 1 \
	   python3 landscape.py \
	   --dataset $ds \
	   --arch $arch \
	   --optim $optim \
	   --batch-size $bs \
	   --range -2.0 2.0 \
	   --model_path $model_path \
	   --pca $pca_path
    bs=$((bs*2))
done
