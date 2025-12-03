#!/bin/bash

# Trainer端环境变量设置脚本
echo "Setting up Trainer environment..."

# Transfer Engine 基本配置
export TRANSFER_ENGINE_TYPE="utrans"
export TRANSFER_ENGINE_META_SERVICE_ADDRESS="127.0.0.1:8081"
export TRANSFER_ENGINE_LOCAL_ADDRESS="33.184.120.50"
export TRANSFER_ENGINE_LOCAL_PORT="8082"

# Transfer Engine Service 配置 - Trainer端
export TRANSFER_ENGINE_SERVICE_ROLE="TRAIN"
export TRANSFER_ENGINE_SERVICE_ADDRESS="33.184.120.50"
export TRANSFER_ENGINE_SERVICE_PORT="8082"
export TRANSFER_ENGINE_PEERS_HOST="33.184.127.58:8081:8080"

# Python路径设置
export PYTHONPATH="${PYTHONPATH}:$(pwd)/build/python"

# 其他可能需要的环境变量
export CUDA_VISIBLE_DEVICES="0"

WORLD_SIZE=${WORLD_SIZE:-1}

echo "Trainer environment variables set:"
echo "  ROLE: $TRANSFER_ENGINE_SERVICE_ROLE"
echo "  ADDRESS: $TRANSFER_ENGINE_SERVICE_ADDRESS:$TRANSFER_ENGINE_SERVICE_PORT"
echo "  PEERS: $TRANSFER_ENGINE_PEERS_HOST"
echo "  LOCAL: $TRANSFER_ENGINE_LOCAL_ADDRESS:$TRANSFER_ENGINE_LOCAL_PORT"
echo "  TYPE: $TRANSFER_ENGINE_TYPE"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  WORLD_SIZE: $WORLD_SIZE"

# 启动多个训练子进程
echo "Starting trainer processes..."
for ((i=0; i<$WORLD_SIZE; i++)); do
    echo "启动训练进程 rank=$i"
    python3 astate/python/test_rdma/train.py "$@" --role_rank $i --role_size $WORLD_SIZE &
done
wait 