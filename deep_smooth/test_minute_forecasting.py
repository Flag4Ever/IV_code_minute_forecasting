import random
import argparse
import os
import pandas as pd
import numpy as np

import torch

from model.deep_model import IvSmoother
from model.prior_model import ATMInterpolator
from dataset.iv_data import IvDataset


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def resolve_flag_column(df: pd.DataFrame, preferred: str | None, test: bool = True) -> str | None:
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend([
        "train_flag_forecast",
        "train_flag_inter",
        "train_flag_extra",
        "train_flag",
    ])
    for column in candidates:
        if column and column in df.columns:
            if preferred and column != preferred:
                flag_value = 0 if test else 1
                print(
                    f"⚠️  未找到列 {preferred}，改用 {column} 作为{'测试' if test else '训练'}标记 (取值 {flag_value})"
                )
            return column
    print("⚠️  数据中未找到任何标记列，将直接使用全部样本")
    return None


def main() -> None:
    seed = 1234
    set_global_seed(seed)

    parser = argparse.ArgumentParser(description="单分钟测试脚本 (跨分钟预测)")
    parser.add_argument("--path_prefix", type=str, default="smooth_forecast")
    parser.add_argument("--option_name", type=str, default="zhongzheng1000_minute")
    parser.add_argument("--split_name", type=str, default="forecast")
    parser.add_argument("--data_type", type=str, default="clear")
    parser.add_argument("--minutes", type=str, default="2025-08-01 09:40:00", help="逗号分隔的测试分钟列表")
    parser.add_argument("--source_minute", type=str, default="2025-08-01 09:30:00", help="模型来源分钟")
    parser.add_argument("--activation", type=str, default="sigmoid")
    parser.add_argument("--model_size_list", type=str, default="[40,40,40,40]")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--w_d", type=float, default=0.0)
    parser.add_argument("--flag_column", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="./checkpoint")
    args = parser.parse_args()

    model_size_list = eval(args.model_size_list)
    target_minutes = [m.strip() for m in args.minutes.split(",") if m.strip()]
    if not target_minutes:
        print("错误: 未指定任何测试分钟")
        raise SystemExit(1)

    # 设备配置
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA 不可用，回退到 CPU")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("WARNING: MPS 不可用，回退到 CPU")
        device = "cpu"
    print(f"使用设备: {device}")

    # 加载数据
    data_path = f"../data/{args.option_name}/{args.option_name}.csv"
    print(f"加载数据: {data_path}")
    df_all = pd.read_csv(data_path)
    df_all["quote_minute"] = pd.to_datetime(df_all["quote_minute"])

    flag_column = resolve_flag_column(df_all, args.flag_column, test=True)

    # 从source_minute获取训练数据，用于创建ATM插值器
    source_minute = pd.to_datetime(args.source_minute)
    df_source = df_all[df_all["quote_minute"] == source_minute].copy()
    if flag_column:
        df_source_train = df_source[df_source[flag_column] == 1].copy()
    else:
        df_source_train = df_source.copy()

    if df_source_train.empty:
        print(f"错误: 源分钟 {args.source_minute} 无训练数据")
        raise SystemExit(1)

    # 创建ATM插值器
    df_source_train = df_source_train.sort_values(by='ttm')
    df_atm = df_source_train.loc[df_source_train.groupby("ttm")["logm"].apply(lambda x: abs(x).idxmin())]
    w_atm_fun = ATMInterpolator(df_atm).interpolate()

    # 初始化模型
    model = IvSmoother(model_size_list, activation=args.activation, prior="svi", w_atm_fun=w_atm_fun)
    model.to(device)

    # 加载模型权重
    minute_token = args.source_minute.replace(" ", "_").replace(":", "_")
    model_dir_path = os.path.join(
        args.output_root,
        args.path_prefix,
        f"{args.option_name}_{args.split_name}_{args.data_type}",
        minute_token,
    )
    model_path = f"{model_dir_path}/seed_{seed}_lr_{args.lr}_w_d_{args.w_d}_best_model.pth"
    print(f"加载模型: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 对每个目标分钟进行测试
    for test_minute in target_minutes:
        quote_minute = pd.to_datetime(test_minute)
        df = df_all[df_all["quote_minute"] == quote_minute].copy()

        # 筛选测试数据
        if flag_column:
            df = df[df[flag_column] == 0].copy()

        if df.empty:
            print(f"⚠️  测试分钟 {test_minute} 无测试样本，跳过")
            continue

        print(f"\n===== 测试分钟 {test_minute} (模型来自 {args.source_minute}) =====")

        # 准备测试数据
        iv_data = IvDataset(df)
        data_di = iv_data.get_ivsmoother_dict()["di"]
        for key in list(data_di.keys()):
            data_di[key] = data_di[key].to(device)

        # 进行测试
        metric = model.get_test_loss(data_di)
        rmse = metric["loss_fit_rmse"].item()
        mape = metric["loss_fit_mape"].item()

        # 输出结果（格式与deep_nn_line保持一致）
        print(
            f"deep_smooth\t{args.path_prefix}\t{args.option_name}\t0.0\t{test_minute}\t{args.source_minute}\t{args.split_name}\t{args.data_type}\t{rmse}\t{mape}"
        )
        print(f"RMSE: {rmse:.6f} | MAPE: {mape:.6f}")

    print("\n所有指定分钟测试完成")


if __name__ == "__main__":
    main()