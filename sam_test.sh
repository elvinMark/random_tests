#!/bin/bash
#YBATCH -r am_4
#SBATCH -N 1
#SBATCH -J sam_test_profiler
#SBATCH --output output/%j.out

# ======== Pyenv/ ========
# export PYENV_ROOT=$HOME/.pyenv
# export PATH=$PYENV_ROOT/bin:$PATH
# eval "$(pyenv init -)"

# ======== Modules ========
. /etc/profile.d/modules.sh
module load cuda/11.1 cudnn/cuda-11.1/8.0 nccl/cuda-11.1/2.7.8 openmpi/3.1.6

export MASTER_ADDR=$(/usr/sbin/ip a show | grep inet | grep 192.168.205 | head -1 | cut -d " " -f 6 | cut -d "/" -f 1)
export MASTER_PORT=3545

export NGPUS=4
export NUM_PROC=4
mpirun -npernode $NUM_PROC -np $NGPUS \
python sam_test.py \
  --dist-url $MASTER_ADDR \
  --epochs 200 \
  --batch-size 8192 \
  --nbs 0.02 \
  --lr 0.2 \
  --experiment bs_8192_ngpus_4_lr_0.2_nbs_0.02
