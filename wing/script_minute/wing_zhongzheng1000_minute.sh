#!/bin/bash

# Wing 分钟级测试脚本
# Wing是理论优化模型，使用CPU进行参数优化，自动并行处理

OPTION_NAME="zhongzheng1000_minute"
DATA_TYPE="clear"
RESULT_FOLDER="./results"

echo "=========================================="
echo "Wing 分钟级测试开始"
echo "数据集: ${OPTION_NAME}"
echo "=========================================="
echo ""

# 定义测试函数（添加时间记录）
run_test() {
    local split_name=$1
    local path_prefix=$2
    local log_file="./script_minute/log/${split_name}_${OPTION_NAME}.log"

    # 创建日志目录
    mkdir -p "$(dirname "$log_file")"

    # 使用大括号包裹命令，确保时间戳和输出都写入日志
    {
        echo "=========================================="
        echo "测试开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "划分: $split_name | 数据集: ${OPTION_NAME}"
        echo "=========================================="
        echo ""

        python test_date_mult_minute.py \
            --option_name ${OPTION_NAME} \
            --split_name "$split_name" \
            --data_type ${DATA_TYPE} \
            --result_folder ${RESULT_FOLDER} \
            --path_prefix "$path_prefix"

        echo ""
        echo "=========================================="
        echo "测试结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
    } 2>&1 | tee "$log_file"

    echo "✅ $split_name 测试完成，日志保存至: $log_file"
    echo ""
}

# Extra 划分测试
run_test "extra" "wing_zhongzheng1000_minute_extra"

# Inter 划分测试
run_test "inter" "wing_zhongzheng1000_minute_inter"

echo "=========================================="
echo "所有测试完成！"
echo "结果文件："
echo "  - ${RESULT_FOLDER}/wing_zhongzheng1000_minute_extra.txt"
echo "  - ${RESULT_FOLDER}/wing_zhongzheng1000_minute_inter.txt"
echo "日志文件："
echo "  - script_minute/log/extra_${OPTION_NAME}.log"
echo "  - script_minute/log/inter_${OPTION_NAME}.log"
echo "=========================================="
