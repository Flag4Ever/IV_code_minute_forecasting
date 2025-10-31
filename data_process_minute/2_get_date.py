# 生成minute_groups.txt便于后续训练
# 适配分钟级数据处理

import pandas as pd


def chunk_list(lst, n):
    """
    将列表分割成每n个元素一组
    """
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def save_to_txt(data, filename):
    """
    将分组数据保存到文本文件
    每行一组，组内元素用空格分隔
    """
    with open(filename, 'w') as file:
        for sublist in data:
            # 将子列表中的元素转换为字符串，并用空格分隔
            line = ' '.join(map(str, sublist))
            # 写入文件并换行
            file.write(line + '\n')


if __name__ == "__main__":
    option_name = "hushen300_minute"
    per_num = 10  # 每组包含的分钟数（可以根据需要调整）

    data_path = f"../data/{option_name}/{option_name}.csv"
    print(f"读取数据: {data_path}")

    data = pd.read_csv(data_path)

    # 转换为datetime类型
    data["quote_minute"] = pd.to_datetime(data["quote_minute"])

    # 获取唯一的分钟时间戳
    minutes = data["quote_minute"].unique()
    print(f"总共 {len(minutes)} 个唯一分钟")

    # 按时间排序
    minutes = sorted(minutes)

    # 转换为字符串格式（便于保存）
    minutes_str = [str(m) for m in minutes]

    # 分组
    minute_groups = chunk_list(minutes_str, per_num)

    # 保存到文件
    output_file = f"../data/{option_name}/minute_groups.txt"
    save_to_txt(minute_groups, output_file)

    print(f"\n生成了 {len(minute_groups)} 个分组")
    print(f"每组最多 {per_num} 个分钟")
    print(f"保存到: {output_file}")

    # 打印前几组作为示例
    print("\n前3组示例:")
    for i, group in enumerate(minute_groups[:3]):
        print(f"  组{i+1}: {group}")

    print("\n2-分钟组生成完成!")
