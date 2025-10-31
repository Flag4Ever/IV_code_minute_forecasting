#!/bin/bash

PATH_PREFIX="sabr_forecast_all"
OPTION_NAME="hushen300_minute"
SPLIT_NAME="forecast"
DATA_TYPE="clear"
FLAG_COLUMN="train_flag_inter"
RESULT_FOLDER="./results_forecasting"
OFFSET_MINUTES=10

mkdir -p "log/${PATH_PREFIX}"
log_file="log/${PATH_PREFIX}/test_shift_${SPLIT_NAME}_${OPTION_NAME}.log"
mkdir -p "$(dirname "$log_file")"

pairs=$(python <<PY
import pandas as pd
from pathlib import Path
from datetime import timedelta

option_name = "hushen300_minute"
flag_column = "train_flag_inter"
offset = int("${OFFSET_MINUTES}")

path = Path("../data") / option_name / f"{option_name}.csv"
df = pd.read_csv(path)
df['quote_minute'] = pd.to_datetime(df['quote_minute'])
all_minutes = set(df['quote_minute'])
if flag_column in df.columns:
    train_minutes = sorted(df[df[flag_column] == 1]['quote_minute'].unique())
else:
    train_minutes = sorted(all_minutes)

pairs = []
for minute in train_minutes:
    target = minute + timedelta(minutes=offset)
    if target in all_minutes:
        pairs.append((minute.strftime("%Y-%m-%d %H:%M:%S"), target.strftime("%Y-%m-%d %H:%M:%S")))

for src, tgt in pairs:
    print(f"{src},{tgt}")
PY
)

if [[ -z "$pairs" ]]; then
    echo "没有找到可用的分钟映射，终止测试" >&2
    exit 1
fi

{
    echo "=========================================="
    echo "SABR跨分钟批量测试开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "模型来源标记列: ${FLAG_COLUMN}"
    echo "测试偏移: ${OFFSET_MINUTES} 分钟"
    echo "=========================================="

    while IFS=',' read -r source_minute target_minute; do
        [[ -z "$source_minute" || -z "$target_minute" ]] && continue
        echo "源分钟: $source_minute -> 测试分钟: $target_minute"
        python test_minute_forecasting.py \
            --option_name ${OPTION_NAME} \
            --split_name ${SPLIT_NAME} \
            --data_type ${DATA_TYPE} \
            --minutes "$target_minute" \
            --source_minute "$source_minute" \
            --flag_column ${FLAG_COLUMN} \
            --result_folder ${RESULT_FOLDER} \
            --path_prefix ${PATH_PREFIX}
    done <<< "$pairs"

    echo ""
    echo "=========================================="
    echo "SABR跨分钟批量测试结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
} >>"${log_file}" 2>&1

echo "✅ SABR批量测试完成，日志: ${log_file}"