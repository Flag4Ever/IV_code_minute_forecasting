import argparse
import os
import pickle
import pandas as pd
import numpy as np
import py_vollib_vectorized
from svi import SVI


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


def load_svi_params(param_file: str) -> dict:
    """加载SVI参数"""
    try:
        with open(param_file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"SVI参数文件未找到: {param_file}")
    except Exception as e:
        raise RuntimeError(f"加载SVI参数失败: {str(e)}")


def test_svi_forecasting(df_test: pd.DataFrame, svi_params: dict, option_name: str,
                        source_minute: str, target_minute: str) -> dict:
    """使用SVI参数对测试数据进行预测"""

    # 准备测试数据格式
    df_test = df_test.copy()
    df_test["r"] = np.sqrt(df_test["ttm"])
    df_test["implied_volatility"] = df_test["iv"]
    df_test["z"] = df_test["logm"] / df_test["r"]

    ex_dates = df_test["expiry"].unique()

    mse_all = 0
    mape_all = 0
    test_num_all = 0
    mse_price_all = 0
    mape_price_all = 0
    but_loss_all = 0
    ex_date_len = 0

    results_by_expiry = {}

    for ex_date in ex_dates:
        data_test = df_test[df_test["expiry"] == ex_date]

        if data_test.empty:
            continue

        # 获取对应的SVI参数
        ex_date_str = str(ex_date)
        if ex_date_str not in svi_params:
            print(f"⚠️  到期日 {ex_date} 无对应的SVI参数，跳过")
            continue

        params = svi_params[ex_date_str]

        try:
            # 使用SVI模型预测隐含波动率
            svi_model = SVI(
                a=params['a'],
                b=params['b'],
                rho=params['rho'],
                sigma=params['sigma'],
                m=params['m']
            )

            iv_fitted = svi_model.implied_volatility(
                data_test["z"].values,
                params['a'], params['b'], params['rho'], params['sigma'], params['m']
            )
            iv_target = data_test["implied_volatility"].values

            # 计算隐含波动率误差
            mse = np.sum((iv_fitted - iv_target) ** 2)
            mape = np.sum(np.abs((iv_fitted - iv_target) / iv_target))
            n_samples = len(iv_target)

            # 计算期权价格误差
            price_target = (data_test["bid"].values + data_test["ask"].values) / 2
            cp_flag = np.where(data_test["z"] > 0, 'c', 'p')

            # Black-Scholes 定价
            price_hat = py_vollib_vectorized.vectorized_black_scholes(
                cp_flag,
                S=data_test["forward"].values,
                K=data_test["strike_price"].values,
                t=data_test["ttm"].values,
                r=np.zeros_like(data_test["ttm"].values),
                sigma=iv_fitted
            ).values.reshape(-1)

            # 折现
            price_hat = data_test["discount_factor"].values * price_hat

            mse_price = np.sum((price_hat - price_target) ** 2)
            mape_price = np.sum(np.abs((price_hat - price_target) / (price_target + 1e-6)))

            # 计算蝶式无套利损失
            base_option_name = option_name.replace("_minute", "")
            if base_option_name in ["hushen300", "zhongzheng1000", "shangzheng50"]:
                z_min, z_max = -0.5, 0.5
            else:
                z_min, z_max = -1.5, 0.5

            arb_z = np.linspace(z_min, z_max, num=100)
            arb_loss = svi_model.get_arb_loss(
                r=data_test["r"].values[0],
                z=arb_z,
                a=params['a'], b=params['b'], rho=params['rho'],
                sigma=params['sigma'], m=params['m']
            )

            but_loss_all += arb_loss["l_butterfly"]

            # 累加统计
            mse_all += mse
            mape_all += mape
            test_num_all += n_samples
            mse_price_all += mse_price
            mape_price_all += mape_price
            ex_date_len += 1

            results_by_expiry[ex_date_str] = {
                'rmse': np.sqrt(mse / n_samples),
                'mape': mape / n_samples,
                'n_samples': n_samples
            }

        except Exception as e:
            print(f"❌ 到期日 {ex_date} 预测失败: {str(e)}")
            continue

    if test_num_all == 0:
        raise ValueError("没有成功预测的样本")

    # 计算总体指标
    rmse = np.sqrt(mse_all / test_num_all)
    mape = mape_all / test_num_all
    price_rmse = np.sqrt(mse_price_all / test_num_all)
    price_mape = mape_price_all / test_num_all
    but_loss = but_loss_all / (100 * ex_date_len) if ex_date_len > 0 else 0

    return {
        'rmse': rmse,
        'mape': mape,
        'price_rmse': price_rmse,
        'price_mape': price_mape,
        'but_loss': but_loss,
        'n_samples': test_num_all,
        'n_expiry_dates': ex_date_len,
        'results_by_expiry': results_by_expiry
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SVI跨分钟预测测试")
    parser.add_argument("--option_name", type=str, default="zhongzheng1000_minute")
    parser.add_argument("--split_name", type=str, default="forecast")
    parser.add_argument("--data_type", type=str, default="clear")
    parser.add_argument("--minutes", type=str, default="2025-08-01 09:40:00", help="逗号分隔的测试分钟列表")
    parser.add_argument("--source_minute", type=str, default="2025-08-01 09:30:00", help="模型来源分钟")
    parser.add_argument("--flag_column", type=str, default=None)
    parser.add_argument("--result_folder", type=str, default="./results_forecasting")
    parser.add_argument("--path_prefix", type=str, default="svi_forecast")
    args = parser.parse_args()

    target_minutes = [m.strip() for m in args.minutes.split(",") if m.strip()]
    if not target_minutes:
        print("错误: 未指定任何测试分钟")
        raise SystemExit(1)

    # 加载数据
    data_path = f"../data/{args.option_name}/{args.option_name}.csv"
    print(f"加载数据: {data_path}")
    df_all = pd.read_csv(data_path)
    df_all["quote_minute"] = pd.to_datetime(df_all["quote_minute"])

    flag_column = resolve_flag_column(df_all, args.flag_column, test=True)

    # 加载SVI参数
    minute_str = args.source_minute.replace(' ', '_').replace(':', '_')
    param_file = f"{args.result_folder}/{args.path_prefix}/{args.option_name}_{args.split_name}_{args.data_type}/svi_params_{minute_str}.pkl"

    print(f"加载SVI参数: {param_file}")
    param_data = load_svi_params(param_file)
    svi_params = param_data['svi_params']

    print(f"成功加载SVI参数，包含 {len(svi_params)} 个到期日")

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

        try:
            result = test_svi_forecasting(
                df, svi_params, args.option_name, args.source_minute, test_minute
            )

            rmse = result['rmse']
            mape = result['mape']
            price_rmse = result['price_rmse']
            price_mape = result['price_mape']
            but_loss = result['but_loss']

            # 输出结果（格式与其他模型保持一致）
            print(
                f"svi\t{args.path_prefix}\t{args.option_name}\t0.0\t{test_minute}\t{args.source_minute}\t{args.split_name}\t{args.data_type}\t{rmse}\t{mape}\t{price_rmse}\t{price_mape}\t0\t{but_loss}"
            )
            print(f"RMSE: {rmse:.6f} | MAPE: {mape:.6f} | Price RMSE: {price_rmse:.6f} | Price MAPE: {price_mape:.6f}")

        except Exception as e:
            print(f"❌ 测试分钟 {test_minute} 失败: {str(e)}")

    print("\n所有指定分钟测试完成")


if __name__ == "__main__":
    main()