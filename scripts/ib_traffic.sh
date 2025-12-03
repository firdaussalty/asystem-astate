#!/bin/bash

# 检查 bc 是否存在
if ! command -v bc &> /dev/null; then
    echo "错误：未找到 bc 工具。请先安装 bc（例如：sudo apt install bc）" >&2
    exit 1
fi

declare -A prev_rx
declare -A prev_tx
declare -A prev_time

while true; do
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    total_rx=0
    total_tx=0

    for dev in $(ls /sys/class/infiniband/); do
        counters_dir="/sys/class/infiniband/$dev/ports/1/counters"
        if [ -d "$counters_dir" ]; then
            rx_file="$counters_dir/port_rcv_data"
            tx_file="$counters_dir/port_xmit_data"

            if [ -f "$rx_file" ] && [ -f "$tx_file" ]; then
                current_rx=$(cat "$rx_file")
                current_tx=$(cat "$tx_file")

                if [[ -n "${prev_rx[$dev]}" ]]; then
                    rx_diff=$((current_rx - prev_rx[$dev]))
                    tx_diff=$((current_tx - prev_tx[$dev]))

                    # 获取毫秒级时间戳（格式：秒.毫秒）
                    start_time=${prev_time[$dev]:-$(date +%s.%3N)}
                    end_time=$(date +%s.%3N)

                    # 计算间隔（单位：秒）
                    interval=$(echo "$end_time - $start_time" | bc)

                    # 避免除以零（最小间隔 1 毫秒 = 0.001 秒）
                    if (( $(echo "$interval <= 0" | bc -l) )); then
                        interval=0.001
                    fi

                    # 转换为 GB/s（考虑时间间隔）
                    # counter: 每个双字（32 位）= 4 字节
                    rx_gbps=$(echo "scale=2; ($rx_diff * 4 / 1073741824) / $interval" | bc)
                    tx_gbps=$(echo "scale=2; ($tx_diff * 4 / 1073741824) / $interval" | bc)

                    # 如果流量非零，打印单个设备信息
                    if (( $(echo "$rx_gbps > 0 || $tx_gbps > 0" | bc -l) )); then
                        echo "$timestamp - $dev RX: $rx_gbps GB/s, TX: $tx_gbps GB/s"
                        #echo "Debug: $dev RX Diff: $rx_diff, Interval: $interval 秒"
                    fi

                    total_rx=$(echo "$total_rx + $rx_gbps" | bc)
                    total_tx=$(echo "$total_tx + $tx_gbps" | bc)
                fi

                prev_rx[$dev]=$current_rx
                prev_tx[$dev]=$current_tx
                prev_time[$dev]="$end_time"  # 存储毫秒级时间戳
            else
                echo "$timestamp - $dev 的 port_rcv_data 或 port_xmit_data 文件不存在，跳过..."
            fi
        else
            echo "$timestamp - $dev 无 counters 目录，跳过..."
        fi
    done

    # 判断总流量是否为 0
    total_non_zero=$(echo "$total_rx > 0 || $total_tx > 0" | bc -l)

    if [ "$total_non_zero" -eq 1 ]; then
        echo "$timestamp - 总流量: RX: $total_rx GB/s, TX: $total_tx GB/s"
        echo ""
    else
        echo "$timestamp - 总流量: RX: 0.0000 GB/s, TX: 0.0000 GB/s"
    fi

    sleep 1
done