import random
import argparse
import os
import pandas as pd
import numpy as np

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

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


def resolve_flag_column(df: pd.DataFrame, preferred: str | None) -> str | None:
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
                print(f"⚠️  未找到列 {preferred}，改用 {column}")
            return column
    print("⚠️  数据中未找到任何训练标记列，将使用全部样本")
    return None


def main() -> None:
    seed = 1234
    set_global_seed(seed)

    parser = argparse.ArgumentParser(description="分钟级训练（forecast 划分）")
    parser.add_argument("--path_prefix", type=str, default="smooth_forecast")
    parser.add_argument("--option_name", type=str, default="zhongzheng1000_minute")
    parser.add_argument("--split_name", type=str, default="forecast")
    parser.add_argument("--data_type", type=str, default="clear")
    parser.add_argument("--quote_minute", type=str, default="2025-08-01 09:30:00")
    parser.add_argument("--n_restart", type=int, default=4)
    parser.add_argument("--activation", type=str, default="sigmoid")
    parser.add_argument("--model_size_list", type=str, default="[40,40,40,40]")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--w_d", type=float, default=0.0)
    parser.add_argument("--flag_column", type=str, default=None)
    args = parser.parse_args()

    # 设备配置
    device = args.device
    if device == "cuda":
        if torch.cuda.is_available():
            print(f"使用设备: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            print("WARNING: CUDA不可用，回退到CPU")
            device = "cpu"
    elif device == "mps":
        if torch.backends.mps.is_available():
            print(f"使用设备: MPS (Apple Silicon)")
        else:
            print("WARNING: MPS不可用，回退到CPU")
            device = "cpu"
    else:
        print(f"使用设备: CPU")

    model_size_list = eval(args.model_size_list)

    # 加载数据
    data_path = f"../data/{args.option_name}/{args.option_name}.csv"
    print(f"加载数据: {data_path}")
    df = pd.read_csv(data_path)
    df["quote_minute"] = pd.to_datetime(df["quote_minute"])
    target_minute = pd.to_datetime(args.quote_minute)

    print(f"筛选分钟: {args.quote_minute}")
    df = df[df["quote_minute"] == target_minute].copy()
    print(f"筛选后数据行数: {len(df)}")
    if df.empty:
        print(f"错误: 未找到分钟 {args.quote_minute} 的数据")
        raise SystemExit(1)

    # 根据标记列筛选训练数据
    flag_column = resolve_flag_column(df, args.flag_column)
    if flag_column:
        df = df[df[flag_column] == 1].copy()
        print(f"按照 {flag_column}=1 筛选后数据行数: {len(df)}")
        if df.empty:
            print("错误: 当前分钟训练样本为空")
            raise SystemExit(1)

    # 数据预处理
    df = df.sort_values(by='ttm')
    df_atm = df.loc[df.groupby("ttm")["logm"].apply(lambda x: abs(x).idxmin())]
    w_atm_fun = ATMInterpolator(df_atm).interpolate()

    iv_data = IvDataset(df)
    data_di = iv_data.get_ivsmoother_dict()["di"]
    for key in list(data_di.keys()):
        data_di[key] = data_di[key].to(device)

    # 模型初始化
    model = IvSmoother(model_size_list, activation=args.activation, prior="svi", w_atm_fun=w_atm_fun)
    model.to(device)

    # 模型编译优化
    enable_compile = (device == "cuda" and torch.cuda.is_available())
    if enable_compile:
        print(f"使用 torch.compile 优化模型 (设备: {device})")
        compile_model = torch.compile(model, mode="max-autotune")
    else:
        print(f"不使用 torch.compile (设备: {device})")
        compile_model = model

    # 模型保存路径
    minute_str = args.quote_minute.replace(' ', '_').replace(':', '_')
    model_dir_path = f"./checkpoint/{args.path_prefix}/{args.option_name}_{args.split_name}_{args.data_type}/{minute_str}/"
    os.makedirs(model_dir_path, exist_ok=True)

    compile_model.train()

    # 训练配置
    optimizer = Adam(compile_model.parameters(), lr=args.lr, weight_decay=args.w_d)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500, threshold=0.01, threshold_mode='rel', verbose=True)

    min_loss = 0.0025
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 2000
    epoch = 0
    best_epoch = 0
    n_restart = args.n_restart

    print(f"开始训练，总轮数: {args.epochs}, Restart次数: {n_restart}")

    while epoch < args.epochs:
        compile_model.zero_grad()
        loss = compile_model.get_train_loss(data_di)

        if loss < best_loss * 0.99 and epoch >= 0:
            best_loss = loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_dir_path + f"seed_{seed}_lr_{args.lr}_w_d_{args.w_d}_best_model.pth")
            print(f"Saved! Epoch {epoch+1} Loss: {loss.item(): .5f};lr: {optimizer.state_dict()['param_groups'][0]['lr']:.5f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        epoch += 1

        if loss < min_loss:
            print(f"Training stopped as loss reached {loss:.4f}")
            break

        if epochs_no_improve >= early_stopping_patience and n_restart > 0 and args.epochs - best_epoch - 500 > 0:
            model.load_state_dict(torch.load(model_dir_path + f"seed_{seed}_lr_{args.lr}_w_d_{args.w_d}_best_model.pth", map_location=device))
            if enable_compile:
                compile_model = torch.compile(model, mode="max-autotune")
            else:
                compile_model = model
            optimizer = Adam(compile_model.parameters(), lr=args.lr, weight_decay=args.w_d)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-5, factor=0.5, patience=500, threshold=0.01, threshold_mode='rel', verbose=True)
            epochs_no_improve = 0
            epoch = best_epoch
            n_restart -= 1
            print(f"Restart!Epoch {epoch+1} Loss: {loss.item(): .5f};lr: {optimizer.state_dict()['param_groups'][0]['lr']:.5f}")

        if (epoch + 1) % 2000 == 0:
            torch.save(model.state_dict(), model_dir_path + f"seed_{seed}_lr_{args.lr}_w_d_{args.w_d}_epoch_{epoch+1}.pth")
            print(f"Saved! Epoch {epoch+1} Loss: {loss.item(): .5f};lr: {optimizer.state_dict()['param_groups'][0]['lr']:.5f}")

    print(f"\n训练完成! 分钟: {args.quote_minute}, 划分方式: {args.split_name}")


if __name__ == "__main__":
    main()