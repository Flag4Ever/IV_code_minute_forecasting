#!/bin/bash

PATH_PREFIX="sabr_forecast_all"
OPTION_NAME="hushen300_minute"
SPLIT_NAME="forecast"
DATA_TYPE="clear"
FLAG_COLUMN="train_flag_inter"
RESULT_FOLDER="./results_forecasting"

mkdir -p "log/${PATH_PREFIX}"

get_train_minutes() {
python <<'PY'
import pandas as pd
from pathlib import Path

option_name = "hushen300_minute"
flag_column = "train_flag_inter"
data_path = Path("../data") / option_name / f"{option_name}.csv"
df = pd.read_csv(data_path)
df["quote_minute"] = pd.to_datetime(df["quote_minute"])
if flag_column in df.columns:
    df = df[df[flag_column] == 1]
minutes = sorted(df["quote_minute"].dropna().unique())
for minute in minutes:
    print(pd.Timestamp(minute).strftime("%Y-%m-%d %H:%M:%S"))
PY
}

while IFS= read -r data_minute; do
    [[ -z "${data_minute}" ]] && continue
    log_minute=${data_minute//[: ]/_}
    log_file="log/${PATH_PREFIX}/${log_minute}_${SPLIT_NAME}_${OPTION_NAME}.log"
    mkdir -p "$(dirname "$log_file")"

    {
        echo "=========================================="
        echo "SABR参数拟合开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "分钟: ${data_minute} | 划分: ${SPLIT_NAME}"
        echo "标记列: ${FLAG_COLUMN}"
        echo "=========================================="

        python train_minute_forecasting.py \
            --option_name ${OPTION_NAME} \
            --split_name ${SPLIT_NAME} \
            --data_type ${DATA_TYPE} \
            --quote_minute "${data_minute}" \
            --flag_column ${FLAG_COLUMN} \
            --result_folder ${RESULT_FOLDER} \
            --path_prefix ${PATH_PREFIX}

        echo ""
        echo "=========================================="
        echo "SABR参数拟合结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
    } >>"${log_file}" 2>&1

    echo "✅ SABR参数拟合完成: ${data_minute} (日志: ${log_file})"
done < <(get_train_minutes)

echo "=========================================="
echo "所有SABR参数拟合任务完成!"
echo "=========================================="