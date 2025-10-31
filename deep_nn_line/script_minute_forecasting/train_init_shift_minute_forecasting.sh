#!/bin/bash

PATH_PREFIX="know_forecast_all"
OPTION_NAME="hushen300_minute"
OPTION_NAME_TRAIN="hushen300_minute"
SPLIT_NAME="forecast"
DATA_PRE="logit"
ACTIVATION="relu"
EPOCHS=20000
FIRST_EPOCHS=1000
LR=0.001
GEN_LR=0.001
W_D=0
IS_LAYERNORM="True"
GEN_MODEL="glow_nn_line"
GEN_PATH="glow_nn_svi_2"
GEN_FLOW=4
GEN_BLOCK=2
GEN_STEPS=500000
GEN_GAMMA=0.1
LINE_SIZE=128
FILTER_SIZE=256
FLAG_COLUMN="train_flag_inter"
DEVICE="cuda"

mkdir -p "log/${PATH_PREFIX}"

get_reference_minute() {
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
if df.empty:
    raise SystemExit(1)
minute = df["quote_minute"].min()
if pd.isna(minute):
    raise SystemExit(1)
print(minute.strftime("%Y-%m-%d %H:%M:%S"))
PY
}

REFERENCE_MINUTE=$(get_reference_minute) || {
    echo "ERROR: 无法确定参考训练分钟" >&2
    exit 1
}

LOG_MINUTE=${REFERENCE_MINUTE//[: ]/_}
LOG_FILE="log/${PATH_PREFIX}/${LOG_MINUTE}_${SPLIT_NAME}_${OPTION_NAME}.log"
mkdir -p "$(dirname "$LOG_FILE")"

{
    echo "=========================================="
    echo "初始化训练开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "分钟: ${REFERENCE_MINUTE} | 划分: ${SPLIT_NAME} | 设备: ${DEVICE}"
    echo "标记列: ${FLAG_COLUMN}"
    echo "=========================================="

    python train_init_minute_forecasting.py \
        --path_prefix ${PATH_PREFIX} \
        --option_name ${OPTION_NAME} \
        --option_name_train ${OPTION_NAME_TRAIN} \
        --split_name ${SPLIT_NAME} \
        --data_type clear \
        --quote_minute "${REFERENCE_MINUTE}" \
        --data_pre ${DATA_PRE} \
        --activation ${ACTIVATION} \
        --data_channel 1 \
        --noise_std 0.001 \
        --model_size_list [40,40,40,40] \
        --device ${DEVICE} \
        --epochs ${EPOCHS} \
        --first_epochs ${FIRST_EPOCHS} \
        --lr ${LR} \
        --gen_lr ${GEN_LR} \
        --w_d ${W_D} \
        --is_layernorm ${IS_LAYERNORM} \
        --gen_model ${GEN_MODEL} \
        --gen_path ${GEN_PATH} \
        --gen_flow ${GEN_FLOW} \
        --gen_block ${GEN_BLOCK} \
        --gen_steps ${GEN_STEPS} \
        --gen_gamma ${GEN_GAMMA} \
        --line_size ${LINE_SIZE} \
        --filter_size ${FILTER_SIZE} \
        --flag_column ${FLAG_COLUMN}

    echo ""
    echo "=========================================="
    echo "初始化训练结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
} >>"${LOG_FILE}" 2>&1

echo "✅ 初始化训练完成，日志: ${LOG_FILE}"
