#!/bin/bash

PATH_PREFIX="smooth_forecast_all"
OPTION_NAME="hushen300_minute"
SPLIT_NAME="forecast"
DATA_TYPE="clear"
ACTIVATION="sigmoid"
EPOCHS=10000
LR=0.01
W_D=0
N_RESTART=4
MODEL_SIZE_LIST="[40,40,40,40]"
FLAG_COLUMN="train_flag_inter"
DEVICE="cuda"

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
        echo "训练开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "分钟: ${data_minute} | 划分: ${SPLIT_NAME} | 设备: ${DEVICE}"
        echo "标记列: ${FLAG_COLUMN}"
        echo "=========================================="

        python train_minute_forecasting.py \
            --path_prefix ${PATH_PREFIX} \
            --option_name ${OPTION_NAME} \
            --split_name ${SPLIT_NAME} \
            --data_type ${DATA_TYPE} \
            --quote_minute "${data_minute}" \
            --activation ${ACTIVATION} \
            --model_size_list ${MODEL_SIZE_LIST} \
            --device ${DEVICE} \
            --epochs ${EPOCHS} \
            --lr ${LR} \
            --w_d ${W_D} \
            --n_restart ${N_RESTART} \
            --flag_column ${FLAG_COLUMN}

        echo ""
        echo "=========================================="
        echo "训练结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
    } >>"${log_file}" 2>&1

    echo "✅ 训练完成: ${data_minute} (日志: ${log_file})"
done < <(get_train_minutes)

echo "=========================================="
echo "所有训练任务完成!"
echo "=========================================="