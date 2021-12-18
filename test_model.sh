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

# Batch Size
bs=128

# model_path="../SAM_tests/trained_models/WRN-28-10_with_original_SAM/model_epochs=150_lr=0.008_momentum=0.9_weight_decay=0.0005_nbs=0.0_global_batch=256_local_batch=32.ckpt"
model_path=""


mpirun -npernode $NUM_PROC -np $NGPUS \
       python test_model.py \
       --dataset Imagenet \
       --arch default_Resnet18 \
       --dist-url $MASTER_ADDR \
       --batch-size $bs \
       --test-dataset "test"
       # --model_path $model_path

