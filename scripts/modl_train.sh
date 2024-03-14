#!/bin/bash -l
#SBATCH --partition=rtx3080         # 指定分区
#SBATCH --gres=gpu:1     # 请求一个 a100 GPU
#SBATCH --time=20:00:00        # 设置运行时间限制
#SBATCH --job-name=modl_train-job   # 作业名称
#SBATCH --output=modl_train.out     # 标准输出文件名
#SBATCH --error=modl_train.err      # 标准错误输出文件名


# unset SLURM_EXPORT_ENV
# module load modl



bash /home/woody/rzku/mlvl127h/MoDL_PyTorch/scripts/train.sh
