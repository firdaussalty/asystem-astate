#!/bin/bash

# Infer端环境变量设置脚本
echo "Setting up Infer environment..."

# Transfer Engine 基本配置
export TRANSFER_ENGINE_TYPE="utrans"
export TRANSFER_ENGINE_META_SERVICE_ADDRESS="127.0.0.1:8081"
export TRANSFER_ENGINE_LOCAL_ADDRESS="33.184.127.58"
export TRANSFER_ENGINE_LOCAL_PORT="8080"

# Transfer Engine Service 配置 - Infer端
export TRANSFER_ENGINE_SERVICE_ROLE="INFER"
export TRANSFER_ENGINE_SERVICE_ADDRESS="33.184.127.58"
export TRANSFER_ENGINE_SERVICE_PORT="8081"
export TRANSFER_ENGINE_PEERS_HOST="33.184.120.50:8082:8083"

# Python路径设置
export PYTHONPATH="${PYTHONPATH}:$(pwd)/build/python"

# 其他可能需要的环境变量
export CUDA_VISIBLE_DEVICES="0"

# 分片配置环境变量
export SHARD_ROWS=${SHARD_ROWS:-1}
export SHARD_COLS=${SHARD_COLS:-1}

WORLD_SIZE=${WORLD_SIZE:-8}

echo "Infer environment variables set:"
echo "  ROLE: $TRANSFER_ENGINE_SERVICE_ROLE"
echo "  ADDRESS: $TRANSFER_ENGINE_SERVICE_ADDRESS:$TRANSFER_ENGINE_SERVICE_PORT"
echo "  PEERS: $TRANSFER_ENGINE_PEERS_HOST"
echo "  LOCAL: $TRANSFER_ENGINE_LOCAL_ADDRESS:$TRANSFER_ENGINE_LOCAL_PORT"
echo "  TYPE: $TRANSFER_ENGINE_TYPE"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  分片配置: ${SHARD_ROWS}行 x ${SHARD_COLS}列"

# 启动多个推理子进程
echo "Starting infer processes..."
for ((i=0; i<$WORLD_SIZE; i++)); do
    echo "启动推理进程 rank=$i"
    python3 astate/python/test_rdma/infer.py "$@" --role_rank $i --role_size $WORLD_SIZE --shard_rows $SHARD_ROWS --shard_cols $SHARD_COLS &
done
wait 