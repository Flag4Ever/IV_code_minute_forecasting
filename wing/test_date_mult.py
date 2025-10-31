import argparse
import os
import time
import pandas as pd
import numpy as np
from wing import ArbitrageFreeWingModel
import multiprocessing
import contextlib
import py_vollib_vectorized

def test_wrapper(args_queue, task):
    """带异常处理的测试函数包装器"""
    try:
        df_train, df_test, option_name, data_type, date, split_name = task
        result = test(df_train, df_test, option_name, data_type, date, split_name)
        args_queue.put(("success", result))
    except Exception as e:
        args_queue.put(("error", f"{date} | {str(e)}"))

# 计算价格方面的误差函数
def loss_price_mse(price_hat,price_target):
    return np.sum((price_hat-price_target)**2)

def loss_price_mape(price_hat,price_target):
    return np.sum(np.abs((price_hat-price_target)/(price_target+1e-6)))

def loss_price_spread(price_hat,price_target,price_ask,price_bid):
    s_ask_bid = price_ask - price_bid
    assert (s_ask_bid>0).all(),"有ask小于bid!"
    loss_spread = np.sum((2*np.abs(price_hat-price_target))/(s_ask_bid))
    return loss_spread

def test(df_train,
         df_test,
         option_name,
         data_type,
         date,
         split_name,
         model_name="wing"):
    # 首先判断一下数据是否合理
    assert (df_train["quote_date"]==date).all() == True
    # assert (df_train["name"]==option_name).all() == True
    assert (df_test["quote_date"]==date).all() == True
    # assert (df_test["name"]==option_name).all() == True

    ex_dates = np.unique(df_train["expiry"])
    # print(ex_dates)
    # 确定wing模型中的默认参数
    dc,uc,dsm,usm = -0.2,0.2,0.5,0.5
    mse_all = 0
    mape_all = 0
    test_num_all = 0
    mse_price_all = 0
    mape_price_all = 0
    spread_price_all = 0
    but_loss_all = 0
    for ex_date in ex_dates:
        wing_model = ArbitrageFreeWingModel()
        data_use = df_train[df_train["expiry"]==ex_date]
        data_test = df_test[df_test["expiry"]==ex_date]
        train_x = data_use["logm"].values
        train_iv = data_use["iv"].values
        test_x = data_test["logm"].values
        test_iv = data_test["iv"].values
        price_target = (data_test["bid"].values+data_test["ask"].values)/2
        price_ask = data_test["ask"].values
        price_bid = data_test["bid"].values
        test_num = len(test_x)
        # print(test_num)
        # 各个样本所占用的重要性相同
        vega = np.ones_like(train_x)
        (sc, pc, cc), vc, loss, arbitrage_free = wing_model.calibrate(train_x,train_iv,vega,
                                                                  dc=dc,uc=uc,dsm=dsm,usm=usm,
                                                                  epsilon=1e-6,
                                                                  epochs=100,show_error=True,
                                                                  use_constraints=True)
        if option_name in ["hushen300","zhongzheng1000","shangzheng50"]:
            z_min = -0.5
            z_max = 0.5
        else:
            z_min = -1.5
            z_max = 0.5
        arb_z = np.linspace(z_min,z_max,num=100)
        arb_k = arb_z*np.sqrt(data_test["ttm"].values[0])
        # 返回的是和值
        but_arb = wing_model.get_but_loss(sc, pc, cc, dc, dsm, uc, usm, arb_k, vc)
        but_loss_all += but_arb["l_butterfly"]

        loss_test = wing_model.loss_test((sc,pc,cc),test_x,test_iv,vc,dc,uc,dsm,usm)
        # 损失需要累计计算
        iv_hat = wing_model.skew(test_x, vc, sc, pc, cc, dc, uc, dsm, usm)
        cp_flag = np.where(test_x > 0, 'c', 'p')
        price_hat = py_vollib_vectorized.vectorized_black_scholes(cp_flag,
                                                              S = data_test["forward"].values,
                                                              K = data_test["strike_price"].values,
                                                              t = data_test["ttm"].values,
                                                              r = np.zeros_like(data_test["ttm"].values),
                                                              sigma = iv_hat).values.reshape(-1)
        price_hat = data_test["discount_factor"].values*price_hat
        mse_price = loss_price_mse(price_hat,price_target=price_target)
        mape_price = loss_price_mape(price_hat,price_target)
        """spread_price = loss_price_spread(price_hat=price_hat,
                                         price_target=price_target,
                                         price_ask=price_ask,
                                         price_bid=price_bid)"""
        mse_price_all += mse_price
        mape_price_all += mape_price
        # 不支持spread,默认为0
        # spread_price_all += spread_price
        mse_all += loss_test["mse"]*test_num
        mape_all += loss_test["mape"]*test_num
        test_num_all += test_num
    rmse = np.sqrt(mse_all/test_num_all)
    mape = mape_all/test_num_all
    price_rmse = np.sqrt(mse_price_all/test_num_all) if test_num_all != 0 else 0
    price_mape = mape_price_all/test_num_all if test_num_all != 0 else 0
    price_spread = spread_price_all/test_num_all if test_num_all != 0 else 0
    but_loss = but_loss_all/test_num_all if test_num_all != 0 else 0
    return f"{model_name}\t{option_name}\t{date}\t{data_type}\t{split_name}\t{rmse:.6f}\t{mape:.6f}\t{price_rmse}\t{price_mape}\t{price_spread}\t{but_loss}\n"

def result_writer(file_name, args_queue, total_tasks):
    """独立的写结果进程"""
    with contextlib.closing(open(file_name, "a", buffering=1)) as f:  # 行缓冲模式
        success_count = 0
        error_count = 0
        while success_count + error_count < total_tasks:
            try:
                msg_type, content = args_queue.get(timeout=3000)  # 5分钟超时
                if msg_type == "success":
                    f.write(content)
                    f.flush()  # 强制立即写入磁盘
                    os.fsync(f.fileno())  # 确保写入物理磁盘
                    print(f"✅ 已完成 {content.split()[2]}")
                    success_count += 1
                elif msg_type == "error":
                    print(f"❌ 错误 {content}")
                    error_count += 1
            except Exception as e:
                print(f"队列获取异常: {str(e)}")
                break
        print(f"\n处理完成: {success_count} 成功, {error_count} 失败")

def main(args):
    df = pd.read_csv(f"../data/{args.option_name}/{args.option_name}.csv")
    
    # 数据筛选
    train_flag = 'train_flag_inter' if args.split_name == "inter" else 'train_flag_extra'
    df_train_all = df[df[train_flag] == 1]
    df_test_all = df[df[train_flag] == 0]
    all_quote_dates = df_train_all["quote_date"].unique()
    
    # 准备任务
    tasks = []
    for date in all_quote_dates:
        tasks.append((
            df_train_all[df_train_all['quote_date'] == date].copy(),
            df_test_all[df_test_all['quote_date'] == date].copy(),
            args.option_name,
            args.data_type,
            date,
            args.split_name
        ))
    
    # 创建结果目录
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)
    
    # 结果文件路径
    file_name = f"{args.result_folder}/{args.path_prefix}.txt"
    
    # 清空已有文件（如果需要保留历史结果可以去掉）
    open(file_name, "w").close()

    # 创建跨进程队列
    manager = multiprocessing.Manager()
    args_queue = manager.Queue()
    
    # 启动写结果进程
    writer_process = multiprocessing.Process(
        target=result_writer,
        args=(file_name, args_queue, len(tasks))
    )
    writer_process.start()
    
    # 创建进程池处理任务
    with multiprocessing.Pool(processes=os.cpu_count()//2) as pool:  # 使用半数CPU核心
        for task in tasks:
            pool.apply_async(
                test_wrapper,
                args=(args_queue, task),
                error_callback=lambda e: print(f"进程错误: {str(e)}")
            )
        pool.close()
        pool.join()
    
    # 等待写结果进程完成
    writer_process.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WING模型实时保存多进程测试")
    parser.add_argument("--option_name", type=str, default="sp500_2022")
    parser.add_argument("--data_type", type=str, default="clear")
    parser.add_argument("--split_name", type=str, default="extra")
    parser.add_argument("--result_folder", type=str, default="./results")
    parser.add_argument("--path_prefix", type=str, default="wing")
    args = parser.parse_args()
    
    main(args)