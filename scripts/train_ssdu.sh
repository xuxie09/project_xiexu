#!/bin/bash -l
#SBATCH --partition=rtx3080         # 指定分区
#SBATCH --gres=gpu:1     # 请求一个 a100 GPU
#SBATCH --time=20:00:00        # 设置运行时间限制
#SBATCH --job-name=modl_train-job   # 作业名称
#SBATCH --output=modl_train.out     # 标准输出文件名
#SBATCH --error=modl_train.err      # 标准错误输出文件名
GPU_NUM=0
TRAIN_CONFIG_YAML="configs/base_ssdu,k=1.yaml"

CUDA_VISIBLE_DEVICES=$GPU_NUM python train.py \
    --config=$TRAIN_CONFIG_YAML \
    --write_image=10 \
    --write_lr=True