#!/bin/bash

# =================================================================
#           Offline DPO Training Launch Script
# =================================================================
# 该脚本用于启动基于 verl 的 offline DPO 训练任务。
# 它会调用我们修改后的 main_dpo.py，并加载 dpo_trainer.yaml 配置。

# 确保脚本出错时立即退出
set -e
set -x

# 先确保没有残留 Ray 实例
ray stop

# 启动单个 Ray 实例，监听固定地址和端口
ray start --head --node-ip-address=127.0.0.1 --port=6379

# 告诉后续进程都连到这个实例
export RAY_ADDRESS=127.0.0.1:6379
# --- 1. 用户可配置变量 ---

# 设置要使用的 GPU (例如 "0,1,2,3")
# VISIBLE_DEVICES="0,1,2,3"
VISIBLE_DEVICES="2,3"
# export WANDB_BASE_URL=https://api.bandw.top
export "WANDB_BASE_URL"="https://api.wandb-cn.top"
export WANDB_API_KEY="b1417e5dca21df59cd256fcb8ee63bb5669c3b53"
export HF_ENDPOINT=https://hf-mirror.com

# 设置每个节点使用的 GPU 数量 (必须与 VISIBLE_DEVICES 中的数量一致)
# NPROC_PER_NODE=4
NPROC_PER_NODE=2

# 设置 Hydra 的配置文件路径和名称
# 我们将使用我们创建的 dpo_trainer.yaml
CONFIG_PATH="config"
CONFIG_NAME="dpo_trainer"


MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TRAIN_PATH="recipe/dpo/data/DeepSeek-R1-Distill-Qwen-1.5B_math_n0_K16_len16384_compressed_think_nothink.jsonl"
VAL_PATH="recipe/dpo/data/math.json"

# --- 2. 环境变量设置 ---

# 设置可见的 GPU 设备
export CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES}
# 启用 Hydra 的完整错误堆栈跟踪，方便调试
export HYDRA_FULL_ERROR=1
# 将当前目录添加到 Python 路径中，以确保能找到 recipe 等模块
export PYTHONPATH=$PYTHONPATH:$(pwd)


# --- 3. 训练启动命令 ---

# 使用 torchrun 启动分布式训练
# 它会调用我们修改后的 main_dpo.py
# --config-path 和 --config-name 会告诉 Hydra 加载我们的 DPO 配置文件
# 后续的 `key=value` 参数会覆盖配置文件中的默认值

echo "Starting Offline DPO training..."
echo "Using Model: ${MODEL_PATH}"
echo "Using Data: ${DATA_PATH}"
echo "Using GPUs: ${VISIBLE_DEVICES}"

torchrun --nproc_per_node=${NPROC_PER_NODE}\
    recipe/dpo/main_dpo.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    data.train_files="[${TRAIN_PATH}]" \
    data.val_files="[${VAL_PATH}]" \
    trainer.n_gpus_per_node=${NPROC_PER_NODE} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    2>&1 | tee dpo_training_log.txt

echo "Offline DPO training finished."

