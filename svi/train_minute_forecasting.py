import argparse
import os
import pickle
import pandas as pd
import numpy as np
from svi import SVI


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


def fit_svi_for_minute(df_train: pd.DataFrame, option_name: str, quote_minute: str) -> dict:
    """为单个分钟的训练数据拟合SVI参数"""

    # 准备SVI所需的数据格式
    df_train = df_train.copy()
    df_train["r"] = np.sqrt(df_train["ttm"])
    df_train["implied_volatility"] = df_train["iv"]
    df_train["z"] = df_train["logm"] / df_train["r"]

    # 按到期日分组拟合
    ex_dates = df_train["expiry"].unique()
    svi_params = {}
    fit_results = {}

    print(f"对分钟 {quote_minute} 拟合SVI参数，共 {len(ex_dates)} 个到期日")

    for ex_date in ex_dates:
        data_use = df_train[df_train["expiry"] == ex_date]

        if data_use.empty:
            print(f"⚠️  到期日 {ex_date} 无训练数据，跳过")
            continue

        try:
            svi_model = SVI()
            svi_model.fit(data_use)

            # 保存参数
            svi_params[str(ex_date)] = {
                'a': svi_model.a,
                'b': svi_model.b,
                'rho': svi_model.rho,
                'sigma': svi_model.sigma,
                'm': svi_model.m
            }

            # 计算拟合效果
            iv_fitted = svi_model.implied_volatility(
                data_use["z"].values,
                svi_model.a, svi_model.b, svi_model.rho, svi_model.sigma, svi_model.m
            )
            iv_target = data_use["implied_volatility"].values

            mse = np.mean((iv_fitted - iv_target) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((iv_fitted - iv_target) / iv_target))

            fit_results[str(ex_date)] = {
                'rmse': rmse,
                'mape': mape,
                'n_samples': len(data_use)
            }

            print(f"  到期日 {ex_date}: RMSE={rmse:.6f}, MAPE={mape:.6f}, 样本数={len(data_use)}")

        except Exception as e:
            print(f"❌ 到期日 {ex_date} 拟合失败: {str(e)}")
            continue

    if not svi_params:
        raise ValueError(f"分钟 {quote_minute} 所有到期日拟合均失败")

    return {
        'minute': quote_minute,
        'svi_params': svi_params,
        'fit_results': fit_results,
        'n_expiry_dates': len(svi_params)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SVI分钟级参数拟合（forecast 划分）")
    parser.add_argument("--option_name", type=str, default="zhongzheng1000_minute")
    parser.add_argument("--split_name", type=str, default="forecast")
    parser.add_argument("--data_type", type=str, default="clear")
    parser.add_argument("--quote_minute", type=str, default="2025-08-01 09:30:00")
    parser.add_argument("--flag_column", type=str, default=None)
    parser.add_argument("--result_folder", type=str, default="./results_forecasting")
    parser.add_argument("--path_prefix", type=str, default="svi_forecast")
    args = parser.parse_args()

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

    # 拟合SVI参数
    try:
        result = fit_svi_for_minute(df, args.option_name, args.quote_minute)

        # 创建保存目录
        minute_str = args.quote_minute.replace(' ', '_').replace(':', '_')
        save_dir = f"{args.result_folder}/{args.path_prefix}/{args.option_name}_{args.split_name}_{args.data_type}"
        os.makedirs(save_dir, exist_ok=True)

        # 保存参数
        param_file = f"{save_dir}/svi_params_{minute_str}.pkl"
        with open(param_file, 'wb') as f:
            pickle.dump(result, f)

        print(f"\n✅ SVI参数拟合完成!")
        print(f"   分钟: {args.quote_minute}")
        print(f"   到期日数量: {result['n_expiry_dates']}")
        print(f"   参数文件: {param_file}")

        # 输出简要统计
        if result['fit_results']:
            all_rmse = [r['rmse'] for r in result['fit_results'].values()]
            all_mape = [r['mape'] for r in result['fit_results'].values()]
            print(f"   平均RMSE: {np.mean(all_rmse):.6f}")
            print(f"   平均MAPE: {np.mean(all_mape):.6f}")

    except Exception as e:
        print(f"❌ SVI拟合失败: {str(e)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()