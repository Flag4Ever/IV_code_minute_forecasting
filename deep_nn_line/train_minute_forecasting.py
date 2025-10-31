import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset.iv_data import IvDataset
from model.gen_model import Glow
from model.imp_model import ImpModel


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
    print("⚠️  数据中未找到任何训练标记列，将使用所有样本")
    return None


def fs_to_d(num: float) -> str:
    formatted = f"{num:.10f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def main() -> None:
    seed = 1234
    set_global_seed(seed)

    parser = argparse.ArgumentParser(description="分钟级训练（forecast 划分）")
    parser.add_argument("--path_prefix", type=str, default="gen_sigmoid")
    parser.add_argument("--option_name", type=str, default="generated_data")
    parser.add_argument("--split_name", type=str, default="forecast")
    parser.add_argument("--data_type", type=str, default="clear")
    parser.add_argument("--quote_minute", type=str, default="2025-08-01 09:30:00")
    parser.add_argument("--data_pre", type=str, default="logit")
    parser.add_argument("--activation", type=str, default="sigmoid")
    parser.add_argument("--data_channel", type=int, default=1)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--model_size_list", type=str, default="[32,32,32]")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--first_epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gen_lr", type=float, default=0.0001)
    parser.add_argument("--w_d", type=float, default=0.0)
    parser.add_argument("--is_layernorm", type=eval, default="True")
    parser.add_argument("--gen_model", type=str, default="")
    parser.add_argument("--gen_path", type=str, default="")
    parser.add_argument("--gen_flow", type=int, default=16)
    parser.add_argument("--gen_block", type=int, default=2)
    parser.add_argument("--gen_steps", type=int, default=200000)
    parser.add_argument("--gen_gamma", type=float, default=0.01)
    parser.add_argument("--line_size", type=int, default=128)
    parser.add_argument("--filter_size", type=int, default=512)
    parser.add_argument("--no_lu", action="store_true")
    parser.add_argument("--affine", action="store_true")
    parser.add_argument("--flag_column", type=str, default=None)
    args = parser.parse_args()

    model_size_list = eval(args.model_size_list)

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

    flag_column = resolve_flag_column(df, args.flag_column)
    if flag_column:
        df = df[df[flag_column] == 1].copy()
        print(f"按照 {flag_column}=1 筛选后数据行数: {len(df)}")
        if df.empty:
            print("错误: 当前分钟训练样本为空")
            raise SystemExit(1)

    device = args.device
    iv_data = IvDataset(df, option_name=args.option_name)
    data_di = iv_data.get_ivsmoother_dict()["di"]
    for key in list(data_di.keys()):
        data_di[key] = data_di[key].to(device)

    model = ImpModel(
        model_size_list,
        activation=args.activation,
        data_channel=args.data_channel,
        noise_std=args.noise_std,
        is_layernorm=args.is_layernorm,
    )
    model.to(device)

    minute_str = args.quote_minute.replace(" ", "_").replace(":", "_")
    model_dir_path = f"./checkpoint/{args.path_prefix}/{args.option_name}_{args.split_name}_{args.data_type}/{minute_str}/"
    os.makedirs(model_dir_path, exist_ok=True)
    init_model_dir_path = f"./checkpoint/{args.path_prefix}/{args.option_name}_{args.split_name}_{args.data_type}/init_model/"
    init_model_path = (
        f"{init_model_dir_path}seed_{seed}_{fs_to_d(args.gen_gamma)}"
        f"_lr_{fs_to_d(args.lr)}_gen_lr_{fs_to_d(args.gen_lr)}_w_d_{args.w_d}_best_model.pth"
    )

    print(f"加载初始化模型: {init_model_path}")
    model.load_state_dict(torch.load(init_model_path, map_location=device))

    enable_compile = device == "cuda" and torch.cuda.is_available()
    if enable_compile:
        print(f"使用 torch.compile 优化模型 (设备: {device})")
        compile_model = torch.compile(model, mode="max-autotune")
    else:
        print(f"不使用 torch.compile (设备: {device})")
        compile_model = model
    compile_model.train()

    gen_model_file = f"../{args.gen_model}/{args.gen_path}/model_{args.gen_steps}.pt"
    glow = Glow(
        args.data_channel,
        args.gen_flow,
        args.gen_block,
        length=args.line_size,
        filter_size=args.filter_size,
        affine=args.affine,
        conv_lu=not args.no_lu,
    )

    if device == "cuda" and torch.cuda.is_available():
        gen_device = device
        print("Glow模型使用 CUDA")
    else:
        gen_device = "cpu"
        print("Glow模型使用 CPU (MPS fallback)")

    glow.to(gen_device)
    print(f"加载Glow模型: {gen_model_file}")
    glow.load_state_dict(torch.load(gen_model_file, map_location=gen_device, weights_only=True))

    if enable_compile and gen_device == "cuda":
        print("使用 torch.compile 优化 Glow 模型")
        glow = torch.compile(glow, mode="max-autotune")

    for param in glow.parameters():
        param.requires_grad = False
    glow.eval()

    optimizer = Adam(compile_model.parameters(), lr=args.lr, weight_decay=args.w_d)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-5, factor=0.5, patience=500, threshold=0.01, threshold_mode='rel', verbose=True)
    min_loss = 0.0001
    best_loss_fit_arb_1 = float('inf')
    best_loss_fit_arb = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 4000

    best_path = (
        f"{model_dir_path}seed_{seed}_{fs_to_d(args.gen_gamma)}"
        f"_lr_{fs_to_d(args.lr)}_gen_lr_{fs_to_d(args.gen_lr)}_w_d_{args.w_d}_best_model.pth"
    )

    if args.option_name in {"hushen300", "shangzheng50", "zhongzheng1000", "zhongzheng1000_minute"}:
        model_logm = torch.linspace(-0.5, 0.5, args.line_size, device=device)
    else:
        model_logm = torch.linspace(-1.5, 0.5, args.line_size, device=device)
    logm_grid, ttm_grid, ttm_num, logm_num = compile_model.gen_iv_grid(
        ttm_s=data_di["ttm_fit:0"].unique(),
        logm_s=model_logm,
    )

    print(f"开始训练，总轮数: {args.epochs}, 第一阶段: {args.first_epochs}")

    for epoch in range(args.epochs):
        if epoch < args.first_epochs:
            loss = model.get_train_loss(data_di)
            loss_gen = torch.tensor([0])
            loss_fit_arb = loss
        elif epoch == args.first_epochs and args.first_epochs != 0:
            model.load_state_dict(torch.load(best_path, map_location=device))
            if enable_compile:
                compile_model = torch.compile(model, mode="max-autotune")
            else:
                compile_model = model
            compile_model.train()
            optimizer = Adam(compile_model.parameters(), lr=args.gen_lr, weight_decay=args.w_d)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-5, factor=0.5, patience=500, threshold=0.01, threshold_mode='rel', verbose=True)
            epochs_no_improve = 0
            loss, loss_gen, loss_fit_arb = compile_model.get_surface_loss(
                data_di,
                feed_model=glow,
                logm_grid=logm_grid,
                ttm_grid=ttm_grid,
                logm_num=args.line_size,
                ttm_num=ttm_num,
                gamma=args.gen_gamma,
                data_pre=args.data_pre,
            )
        else:
            loss, loss_gen, loss_fit_arb = compile_model.get_surface_loss(
                data_di,
                feed_model=glow,
                logm_grid=logm_grid,
                ttm_grid=ttm_grid,
                logm_num=args.line_size,
                ttm_num=ttm_num,
                gamma=args.gen_gamma,
                data_pre=args.data_pre,
            )

        if epoch < args.first_epochs and loss_fit_arb < best_loss_fit_arb_1 * 0.99:
            best_loss_fit_arb_1 = loss_fit_arb
            torch.save(model.state_dict(), best_path)
            print(f"Save!Epoch {epoch+1} Loss: {loss.item(): .5f};Gen: {loss_gen.item(): .5f};fit arb: {loss_fit_arb.item(): .5f};lr: {optimizer.state_dict()['param_groups'][0]['lr']:.5f}; no_improve: {epochs_no_improve}")
            epochs_no_improve = 0
        if loss_gen < -3.0 and loss_fit_arb < best_loss_fit_arb * 0.99:
            best_loss_fit_arb = loss_fit_arb
            torch.save(model.state_dict(), best_path)
            print(f"Save!Epoch {epoch+1} Loss: {loss.item(): .5f};Gen: {loss_gen.item(): .5f};fit arb: {loss_fit_arb.item(): .5f};lr: {optimizer.state_dict()['param_groups'][0]['lr']:.5f}; no_improve: {epochs_no_improve}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if loss_fit_arb < min_loss:
            print(f"Training stopped as loss reached {loss:.4f}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(compile_model.parameters(), max_norm=5)
        optimizer.step()
        scheduler.step(loss)

        if epochs_no_improve >= early_stopping_patience:
            model.load_state_dict(torch.load(best_path, map_location=device))
            if enable_compile:
                compile_model = torch.compile(model, mode="max-autotune")
            else:
                compile_model = model
            compile_model.train()
            new_lr = args.lr if epoch < args.first_epochs else args.gen_lr
            optimizer = Adam(compile_model.parameters(), lr=new_lr, weight_decay=args.w_d)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-5, factor=0.5, patience=500, threshold=0.01, threshold_mode='rel', verbose=True)
            epochs_no_improve = 0
            print(f"Restart!Epoch {epoch+1} Loss: {loss.item(): .5f};Gen: {loss_gen.item(): .5f};fit arb: {loss_fit_arb.item(): .5f};lr: {optimizer.state_dict()['param_groups'][0]['lr']:.5f};no_improve: {epochs_no_improve}")

        if (epoch + 1) % 1000 == 0:
            torch.save(model.state_dict(), model_dir_path + f"seed_{seed}_{fs_to_d(args.gen_gamma)}_lr_{fs_to_d(args.lr)}_gen_lr_{fs_to_d(args.gen_lr)}_w_d_{args.w_d}_epoch_{epoch+1}.pth")
        if (epoch + 1) % 1000 == 0 or loss_gen > 0.1:
            print(f"Epoch {epoch+1} Loss: {loss.item(): .5f};Gen: {loss_gen.item(): .5f};fit arb: {loss_fit_arb.item(): .5f};lr: {optimizer.state_dict()['param_groups'][0]['lr']:.5f};no_improve: {epochs_no_improve}")

    print(f"\n训练完成! 分钟: {args.quote_minute}, 划分方式: {args.split_name}")


if __name__ == "__main__":
    main()
