"""
按分钟处理中证1000期权数据
处理8月1日的9:30, 9:35, 9:40, 9:45四个时间点
保留所有半秒级数据,按分钟分组计算forward等指标
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
from numba.cuda import Out
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils_data_process import extract_forward
from py_vollib_vectorized import vectorized_implied_volatility


def parse_option_contract(symbol):
    """
    解析期权合约代码
    例如: MO2508-C-6000
    返回: (到期年月, 看涨看跌标志, 行权价)
    """
    parts = symbol.split("-")
    if len(parts) == 3:
        contract_month = parts[0]  # MO2508
        cp_flag = parts[1]  # C or P
        strike_price = int(parts[2])  # 6000
        return contract_month, cp_flag, strike_price
    return None, None, None


def get_expiry_date(contract_month):
    """
    根据合约月份获取到期日
    MO2508 表示2025年8月到期
    假设到期日为当月第三个星期五(这是标准的期权到期日规则)
    """
    year = int("20" + contract_month[2:4])  # 25 -> 2025
    month = int(contract_month[4:6])  # 08 -> 8

    # 找到当月第三个星期五
    first_day = pd.Timestamp(year=year, month=month, day=1)
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + pd.Timedelta(days=days_until_friday)
    third_friday = first_friday + pd.Timedelta(weeks=2)

    return third_friday


def load_minute_data(data_dir, target_minutes):
    """
    加载所有期权合约在指定分钟内的所有半秒级数据

    参数:
        data_dir: 数据目录路径
        target_minutes: 目标分钟列表,例如 ['09:30', '09:35', '09:40', '09:45']

    返回:
        DataFrame: 合并后的期权数据
    """
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    data_list = []

    for file_name in tqdm(all_files, desc="加载期权合约数据"):
        file_path = os.path.join(data_dir, file_name)

        # 从文件名解析合约信息
        symbol = file_name.replace(".csv", "")
        contract_month, cp_flag, strike_price = parse_option_contract(symbol)

        if contract_month is None:
            continue

        # 读取数据
        df = pd.read_csv(file_path)

        # 转换时间列
        df["tradingtime"] = pd.to_datetime(df["tradingtime"])
        df["tradingdate"] = pd.to_datetime(df["tradingdate"])

        # 提取分钟部分 (HH:MM格式)
        df["minute_str"] = df["tradingtime"].dt.strftime("%H:%M")

        # 筛选目标分钟内的所有数据
        df_filtered = df[df["minute_str"].isin(target_minutes)].copy()

        if len(df_filtered) == 0:
            continue

        # 添加合约信息
        df_filtered["symbol"] = symbol
        df_filtered["cp_flag"] = cp_flag
        df_filtered["strike_price"] = strike_price
        df_filtered["contract_month"] = contract_month

        # 重命名列
        df_filtered.rename(
            columns={
                "tradingdate": "date",
                "tradingtime": "datetime",
                "lastprice": "option_price",
                "buyprice01": "bid",
                "sellprice01": "ask",
            },
            inplace=True,
        )

        data_list.append(df_filtered)

    # 合并所有数据
    if len(data_list) == 0:
        return pd.DataFrame()

    option_data = pd.concat(data_list, axis=0, ignore_index=True)

    return option_data


def load_second_data(data_dir, target_seconds, tolerance=None):
    """
    加载所有期权合约在指定秒附近的快照数据。

    参数:
        data_dir: 数据目录路径
        target_seconds: 目标秒列表，例如 ['09:30:00', '09:45:00']
                        可以是字符串、pd.Timestamp 或 pd.Timedelta 类型
        tolerance: (可选) 最大允许误差，超过该误差的记录将被丢弃。
                   可以是字符串 (如 '5s') 或 pd.Timedelta。默认不限。

    返回:
        DataFrame: 合并后的期权数据
    """
    if not target_seconds:
        return pd.DataFrame()

    # 统一处理目标秒和容差
    target_seconds_parsed = []
    for ts in target_seconds:
        if isinstance(ts, pd.Timestamp):
            ts_time = ts.time()
            label = ts_time.strftime("%H:%M:%S")
            target_seconds_parsed.append((label, pd.to_timedelta(ts_time.isoformat())))
        elif isinstance(ts, pd.Timedelta):
            label = str(ts)
            target_seconds_parsed.append((label, ts))
        else:
            ts_str = str(ts)
            target_seconds_parsed.append((ts_str, pd.to_timedelta(ts_str)))

    tolerance_td = None
    if tolerance is not None:
        tolerance_td = pd.to_timedelta(tolerance)

    all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    data_list = []

    for file_name in tqdm(all_files, desc="按秒加载期权合约数据"):
        file_path = os.path.join(data_dir, file_name)
        symbol = file_name.replace(".csv", "")
        contract_month, cp_flag, strike_price = parse_option_contract(symbol)

        if contract_month is None:
            continue

        df = pd.read_csv(file_path)
        if df.empty:
            continue

        df["tradingtime"] = pd.to_datetime(df["tradingtime"])
        df["tradingdate"] = pd.to_datetime(df["tradingdate"])
        # 记录分钟和秒字符串，便于后续调试/统计
        df["minute_str"] = df["tradingtime"].dt.strftime("%H:%M")
        df["second_str"] = (
            df["tradingtime"].dt.strftime("%H:%M:%S.%f").str.rstrip("0").str.rstrip(".")
        )

        # 将时间转换为当天起点的 Timedelta，便于计算与目标秒的距离
        df["time_offset"] = df["tradingtime"] - df["tradingtime"].dt.normalize()

        for target_label, target_delta in target_seconds_parsed:
            # 与目标秒的“方向性”差值（Timedelta）：>0 表示在目标之后，<0 表示在目标之前
            delta = df["time_offset"] - target_delta
            abs_diff = delta.abs()

            # 按交易日逐组挑选：优先选择“>= 目标秒”的最近一条，否则回退到“绝对就近”
            def pick_idx(g):
                d = g["time_offset"] - target_delta
                future = d[d >= pd.Timedelta(0)]
                if not future.empty:
                    return future.idxmin()          # 未来方向最近
                else:
                    return d.abs().idxmin()         # 当天没有未来方向→回退为绝对就近

            nearest_idx = df.groupby("tradingdate", group_keys=False).apply(pick_idx)

            if nearest_idx.empty:
                continue

            selected = df.loc[nearest_idx.values].copy()
            selected["symbol"] = symbol
            selected["cp_flag"] = cp_flag
            selected["strike_price"] = strike_price
            selected["contract_month"] = contract_month
            selected["target_second"] = target_label
            # 记录与目标秒的绝对距离，便于后续容差过滤/诊断
            selected["distance_to_target"] = abs_diff.loc[nearest_idx.values].values

            if tolerance_td is not None:
                selected = selected[selected["distance_to_target"] <= tolerance_td]
                if selected.empty:
                    continue

            data_list.append(selected)

    if not data_list:
        return pd.DataFrame()

    option_data = pd.concat(data_list, axis=0, ignore_index=True)

    # 重命名列以保持与分钟加载函数一致
    option_data.rename(
        columns={
            "tradingdate": "date",
            "tradingtime": "datetime",
            "lastprice": "option_price",
            "buyprice01": "bid",
            "sellprice01": "ask",
        },
        inplace=True,
        errors="ignore",
    )
    # 新增：用目标秒直接生成 quote_minute 列，避免被实际时间影响
    option_data["quote_minute"] = (
        option_data["date"].dt.normalize()
        + pd.to_timedelta(option_data["target_second"])
    ).dt.floor("min")
    return option_data


def process_minute_data(option_data, index_data, option_name="zhongzheng1000"):
    """
    处理分钟级别的期权数据
    保留所有半秒级数据,但按分钟分组计算forward和discount_factor

    参数:
        option_data: 期权数据(半秒级)
        index_data: 指数数据(分钟级别)

    返回:
        处理后的数据(保留所有半秒级记录)
    """
    # 获取到期日
    option_data["exdate"] = option_data["contract_month"].apply(get_expiry_date)

    # 合并指数数据
    option_data = pd.merge(
        option_data,
        index_data,
        left_on="quote_minute",
        right_on="datetime",
        how="left",
        suffixes=("", "_index"),
    )

    # 去掉当天到期和明天到期的期权
    option_data = option_data[option_data["date"] != option_data["exdate"]].reset_index(
        drop=True
    )
    option_data = option_data[
        (option_data["date"] + pd.Timedelta(days=1)) != option_data["exdate"]
    ].reset_index(drop=True)

    # 计算到期时间 - 基于分钟级别的时间（年化）
    # 使用quote_minute而不是datetime，确保同一分钟内的数据ttm相同
    option_data["ttm"] = option_data["exdate"] - option_data["quote_minute"]
    option_data["ttm"] = option_data["ttm"].apply(
        lambda x: x.total_seconds() / 60 / (365.25 * 24 * 60)
    )  # 分钟数/年分钟数

    # ===== 按分钟分组计算forward和discount_factor =====
    # 为每个分钟+到期日组合计算一次forward

    # 准备看涨看跌期权配对数据 - 使用每条记录
    option_data_c = option_data[option_data["cp_flag"] == "C"].copy()
    option_data_p = option_data[option_data["cp_flag"] == "P"][
        [
            "datetime",
            "exdate",
            "strike_price",
            "option_price",
            "quote_minute",
            "index_close",
        ]
    ].copy()

    # 按照datetime(精确到半秒)配对
    option_data_cp = pd.merge(
        option_data_c,
        option_data_p,
        on=["datetime", "exdate", "strike_price"],
        suffixes=("_c", "_p"),
    )

    # 按分钟+到期日分组计算forward
    # 对每个分钟内的数据,使用该分钟内所有半秒数据的平均值来计算forward
    data_groups = option_data_cp.groupby(["quote_minute_c", "exdate"])
    data_f_list = []

    for (minute, ex), group in tqdm(
        data_groups, desc="计算远期价格(forward)", total=len(data_groups)
    ):
        # 准备数据格式以适配extract_forward函数
        group_prepared = group.copy()
        group_prepared["date"] = group_prepared["quote_minute_c"]
        group_prepared["c_close"] = group_prepared["option_price_c"]
        group_prepared["p_close"] = group_prepared["option_price_p"]
        # index_close字段应该已经存在,确保使用正确的列名
        if "index_close_c" in group_prepared.columns:
            group_prepared["index_close"] = group_prepared["index_close_c"]

        # 计算ttm - 使用分钟级别的时间（年化）
        group_prepared["ttm"] = (
            group_prepared["exdate"] - group_prepared["quote_minute_c"]
        ).apply(
            lambda x: x.total_seconds() / 60 / (365.25 * 24 * 60)  # 分钟数/年分钟数
        )

        try:
            data_f = extract_forward(group_prepared.reset_index(drop=True))
            data_f["quote_minute"] = minute
            data_f_list.append(data_f)
        except Exception as e:
            print(f"Warning: Failed to extract forward for {minute}, {ex}: {e}")
            continue

    if len(data_f_list) == 0:
        print("Warning: No forward prices calculated")
        return pd.DataFrame()

    data_forward = pd.concat(data_f_list, axis=0)

    # 重命名date列为quote_minute_forward (避免重复列名)
    data_forward = data_forward.rename(columns={"date": "quote_minute_forward"})
    data_forward = data_forward[
        ["quote_minute_forward", "exdate", "forward", "discount_factor"]
    ]

    # 与原始option_data合并 - 按分钟匹配
    option_data = pd.merge(
        option_data,
        data_forward,
        left_on=["quote_minute", "exdate"],
        right_on=["quote_minute_forward", "exdate"],
        how="left",
    )

    # 删除临时列
    option_data = option_data.drop(columns=["quote_minute_forward"])

    # 计算logm
    option_data["logm"] = np.log(option_data["strike_price"] / option_data["forward"])

    # 挑出虚值期权
    mask = ((option_data["cp_flag"] == "P") & (option_data["logm"] <= 0)) | (
        (option_data["cp_flag"] == "C") & (option_data["logm"] > 0)
    )
    option_data = option_data[mask].reset_index(drop=True)

    # 创建put和call列
    option_data["put"] = np.where(
        option_data["cp_flag"] == "P", option_data["option_price"], "NA"
    )
    option_data["call"] = np.where(
        option_data["cp_flag"] == "C", option_data["option_price"], "NA"
    )

    option_data["name"] = option_name

    # 计算隐含波动率 - 对每条半秒级数据计算
    S = option_data["forward"].values
    K = option_data["strike_price"].values
    t = option_data["ttm"].values
    r = np.zeros_like(t)

    flag = np.full(r.shape, fill_value="c")
    flag[option_data["logm"] <= 0] = "p"

    print("计算隐含波动率...")
    try:
        option_data["iv"] = vectorized_implied_volatility(
            option_data["option_price"].values, S, K, t, r, flag, return_as="array"
        )
    except Exception as e:
        print(f"Warning: IV calculation failed: {e}")
        option_data["iv"] = np.nan
    print("隐含波动率计算完成")

    # 计算其他指标
    option_data["w"] = option_data["iv"] ** 2 * option_data["ttm"]
    option_data["expiry"] = option_data["exdate"]
    option_data["m"] = option_data["strike_price"] / option_data["forward"]
    option_data["m_check"] = np.exp(option_data["logm"])

    # 选择需要的列
    option_data = option_data[
        [
            "ttm",
            "logm",
            "put",
            "call",
            "name",
            "bid",
            "ask",
            "iv",
            "w",
            "quote_minute",
            "expiry",
            "m",
            "strike_price",
            "discount_factor",
            "forward",
        ]
    ]

    # 按照quote_minute, expiry, m, ttm排序 (ttm区分半秒级数据)
    option_data = option_data.sort_values(["quote_minute", "expiry", "m", "ttm"])

    return option_data


def date_minute(
    option_data_dir="../数据/202508中证1000期权",
    target_minutes=["09:30", "09:35", "09:40", "09:45"],
):
    # 指定处理的日期
    target_date = pd.Timestamp("2025-08-01")

    print("开始加载期权数据...")
    print(f"目标分钟: {target_minutes}")
    option_data = load_minute_data(option_data_dir, target_minutes)

    if len(option_data) == 0:
        print("错误: 未加载到任何数据!")
        exit(1)

    # 筛选8月1日的数据
    option_data = option_data[option_data["date"].dt.date == target_date.date()].copy()

    print(f"加载了 {len(option_data)} 条半秒级期权记录")
    print(f"分钟分布:\n{option_data['minute_str'].value_counts().sort_index()}")
    print(f"合约数量: {option_data['symbol'].nunique()}")

    # 读取真实的中证1000指数数据
    print("\n加载中证1000指数数据...")
    index_file = "../数据/202508中证1000股指/202508中证1000股指.csv"
    index_raw = pd.read_csv(index_file)
    index_raw["tradingtime"] = pd.to_datetime(index_raw["tradingtime"])

    # 提取分钟部分并向下取整
    index_raw["minute"] = index_raw["tradingtime"].dt.floor("min")

    # 筛选目标日期和目标分钟
    index_filtered = index_raw[
        (index_raw["tradingtime"].dt.date == target_date.date())
        & (index_raw["tradingtime"].dt.strftime("%H:%M").isin(target_minutes))
    ].copy()

    # 按分钟分组，取每分钟的平均价格作为该分钟的指数价格
    index_data = (
        index_filtered.groupby("minute").agg({"index_price": "mean"}).reset_index()
    )
    index_data.rename(
        columns={"minute": "datetime", "index_price": "index_close"}, inplace=True
    )

    print(f"加载了 {len(index_data)} 个分钟的指数数据")
    print(
        f"指数价格范围: {index_data['index_close'].min():.2f} - {index_data['index_close'].max():.2f}"
    )

    print("\n开始处理期权数据...")
    processed_data = process_minute_data(option_data, index_data)
    processed_data = processed_data.drop_duplicates(keep="first")
    if len(processed_data) == 0:
        print("错误: 数据处理失败!")
        exit(1)

    print(f"处理后数据行数: {len(processed_data)}")
    print(f"\n数据预览:")
    print(processed_data.head(20))

    # 保存结果
    output_dir = "../data/zhongzheng1000_minute/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "zhongzheng1000_minute_test.csv")

    processed_data.to_csv(output_file, index=False)
    print(f"\n数据已保存到: {output_file}")

    # 打印统计信息
    print("\n统计信息:")
    print(f"分钟数量: {processed_data['quote_minute'].nunique()}")
    print(f"到期日数量: {processed_data['expiry'].nunique()}")
    print(f"每个分钟的记录数:")
    for minute in processed_data["quote_minute"].unique():
        count = len(processed_data[processed_data["quote_minute"] == minute])
        print(f"  {minute}: {count} 条记录")
    print(f"\nIV统计:\n{processed_data['iv'].describe()}")

    print("\n数据处理完成!")


def date_seconds(
    option_data_dir="../数据/202508中证1000期权",
    index_file="../数据/202508中证1000股指/202508中证1000股指.csv",
    target_seconds=None,
    target_dates=None,
    output_dir="../data/zhongzheng1000_minute/",
    output_file_name="zhongzheng1000_minute.csv",
    tolerance=None,
    type="train",
):
    """
    按秒提取多天的期权与指数数据并输出处理结果。

    参数:
        option_data_dir: 期权原始数据目录
        index_file: 指数数据文件
        target_seconds: 需要提取的秒级时间（列表）
        target_dates: 需要处理的日期（列表或单个值）
        output_dir/output_file_name: 输出路径设置
        tolerance: 可接受的时间差（传入形如 '500ms' 或 pd.Timedelta），默认不限
    """

    if target_seconds is None:
        target_seconds = ["09:30:00"]

    if target_dates is None:
        target_dates = [pd.Timestamp("2025-08-01")]

    # 统一日期格式到当天 00:00，便于比较
    target_dates = [
        pd.Timestamp(d).normalize()
        for d in (
            target_dates
            if isinstance(target_dates, (list, tuple, set))
            else [target_dates]
        )
    ]
    target_dates_set = set(target_dates)

    print("开始加载期权数据...")
    print(f"目标秒: {target_seconds}")
    print(f"目标日期: {[d.strftime('%Y-%m-%d') for d in target_dates]}")
    option_data = load_second_data(option_data_dir, target_seconds, tolerance=tolerance)

    if option_data.empty:
        print("错误: 未加载到任何数据!")
        exit(1)

    # 过滤目标日期
    option_data = option_data[
        option_data["date"].dt.normalize().isin(target_dates_set)
    ].copy()

    if option_data.empty:
        print("错误: 指定日期内无可用期权数据!")
        exit(1)

    print(f"加载了 {len(option_data)} 条期权快照记录")
    print("按目标秒分布:")
    print(option_data["target_second"].value_counts().sort_index())
    print("按日期分布:")
    print(option_data["date"].dt.date.value_counts().sort_index())
    print(f"合约数量: {option_data['symbol'].nunique()}")

    # 读取指数数据
    print("\n加载指数价格数据...")
    index_raw = pd.read_csv(index_file)
    index_raw["tradingtime"] = pd.to_datetime(index_raw["tradingtime"])
    index_raw["second"] = index_raw["tradingtime"].dt.floor("S")

    # 仅保留目标日期与时间
    index_filtered = index_raw[
        index_raw["tradingtime"].dt.normalize().isin(target_dates_set)
        & index_raw["tradingtime"].dt.strftime("%H:%M:%S").isin(target_seconds)
    ].copy()

    if index_filtered.empty:
        print("错误: 指数数据中不存在匹配的日期或时间!")
        exit(1)

    # 指数数据按分钟取平均
    index_data = (
        index_filtered.groupby("second").agg({"index_price": "mean"}).reset_index()
    )
    index_data.rename(
        columns={"second": "datetime", "index_price": "index_close"}, inplace=True
    )

    print(f"加载了 {len(index_data)} 个分钟的指数数据")
    print(
        f"指数价格范围: {index_data['index_close'].min():.2f} - {index_data['index_close'].max():.2f}"
    )

    print("\n开始处理期权数据...")
    processed_data = process_minute_data(option_data, index_data)
    processed_data = processed_data.drop_duplicates(keep="first")
    if processed_data.empty:
        print("错误: 数据处理失败!")
        exit(1)

    print(f"处理后数据行数: {len(processed_data)}")
    print("样本日期列表:")
    print(sorted(processed_data["quote_minute"].dt.date.unique()))
    print("\n数据预览:")
    print(processed_data.head(20))

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_file_name)
    output_file = output_file.replace(".csv", f"_{type}.csv")
    processed_data.to_csv(output_file, index=False)
    print(f"\n{type} 数据已保存到: {output_file}")

    # 打印统计信息
    print("\n统计信息:")
    print(f"分钟数量: {processed_data['quote_minute'].nunique()}")
    print(f"到期日数量: {processed_data['expiry'].nunique()}")
    print("每个分钟的记录数:")
    for minute in processed_data["quote_minute"].unique():
        count = len(processed_data[processed_data["quote_minute"] == minute])
        print(f"  {minute}: {count} 条记录")
    print(f"\nIV统计:\n{processed_data['iv'].describe()}")

    print("\n数据处理完成!")


if __name__ == "__main__":
    target_seconds = [
        "09:30:00",
        "10:00:00",
        "10:30:00",
        "11:00:00",
        "11:20:00",
        "13:00:00",
        "13:30:00",
        "14:00:00",
        "14:30:00",
        "14:50:00",
    ]
    shifted_seconds = [
        "09:40:00",
        "10:10:00",
        "10:40:00",
        "11:10:00",
        "11:30:00",
        "13:10:00",
        "13:40:00",
        "14:10:00",
        "14:40:00",
        "15:00:00",
    ]
    target_dates = [
        "2025-08-04",
        "2025-08-06",
        "2025-08-08",
        "2025-08-11",
        "2025-08-13",
        "2025-08-15",
        "2025-08-18",
        "2025-08-20",
        "2025-08-22",
        "2025-08-25",
        "2025-08-27",
        "2025-08-29",
    ]
    date_seconds(
        option_data_dir="../数据/202508沪深300期权",
        index_file="../数据/202508沪深300股指/202508沪深300股指.csv",
        target_seconds=target_seconds,
        target_dates=target_dates,
        output_dir="../data/hushen300_minute/",
        output_file_name="hushen300_minute.csv",
        tolerance=None,
        type="train"
    )
    date_seconds(option_data_dir="../数据/202508沪深300期权",
        index_file="../数据/202508沪深300股指/202508沪深300股指.csv",
        target_seconds=shifted_seconds,
        target_dates=target_dates,
        output_dir="../data/hushen300_minute/",
        output_file_name="hushen300_minute.csv",
        tolerance=None,
        type="test"
    )