#!/bin/bash
#YBATCH -r any_1
#SBATCH -N 1
#SBATCH -J test
#SBATCH --output output/%j.out


for model_path in $(cat list_models.txt)
do
    echo $model_path
done
