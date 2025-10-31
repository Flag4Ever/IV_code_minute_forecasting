import os
import argparse

import pandas as pd
import numpy as np
import random
from wing import ArbitrageFreeWingModel
import matplotlib.pyplot as plt

def pic(df_train,
        df_test,
        option_name,
        data_type,
        date,
        split_type,
        model_name="sabr",
        pic_folder="./results",
        path_prefix="0826"):
    # 首先判断一下数据是否合理
    assert (df_train["quote_date"]==date).all() == True
    assert (df_train["name"]==option_name).all() == True
    assert (df_test["quote_date"]==date).all() == True
    assert (df_test["name"]==option_name).all() == True

    ex_dates = np.unique(df_train["expiry"])
    # print(ex_dates)
    # 确定wing模型中的默认参数
    dc,uc,dsm,usm = -0.2,0.2,0.5,0.5
    for ex_date in ex_dates:
        wing_model = ArbitrageFreeWingModel()
        data_use = df_train[df_train["expiry"]==ex_date]
        data_test = df_test[df_test["expiry"]==ex_date]
        train_x = data_use["logm"].values
        train_iv = data_use["iv"].values
        test_x = data_test["logm"].values
        test_iv = data_test["iv"].values
        # print(test_num)
        # 各个样本所占用的重要性相同
        vega = np.ones_like(train_x)
        (sc, pc, cc), vc, loss, arbitrage_free = wing_model.calibrate(train_x,train_iv,vega,
                                                                  dc=dc,uc=uc,dsm=dsm,usm=usm,
                                                                  epsilon=1e-6,
                                                                  epochs=100,show_error=True,
                                                                  use_constraints=True)
        pic_moneyness = np.arange(-0.5, 0.5, 0.001)
        pic_iv = wing_model.skew(pic_moneyness, vc, sc, pc, cc, dc, uc, dsm,usm)
        plt.scatter(train_x,train_iv,color="red",marker=".",label="train true")
        plt.scatter(test_x,test_iv,color="blue",marker="*",label="test true")
        plt.plot(pic_moneyness,pic_iv,color="blue",label="pred")
        plt.title(ex_date)
        plt.xlabel("logmoneyness")
        plt.ylabel("iv")
        plt.grid(True)
        if not os.path.exists(f"{pic_folder}/{path_prefix}_{model_name}_{date}"):
            os.makedirs(f"{pic_folder}/{path_prefix}_{model_name}_{date}")
        plt.savefig(f"{pic_folder}/{path_prefix}_{model_name}_{date}/{split_type}_{data_type}_expiry_{ex_date}.png")
        plt.show()

parse = argparse.ArgumentParser(description="info of wing model")
parse.add_argument("--option_name",type=str,default="hushen300",help="期权种类")
parse.add_argument("--data_type",type=str,default="clear")
parse.add_argument("--split_type",type=str,default="extra")
parse.add_argument("--pic_folder",type=str,default="./results")
parse.add_argument("--path_prefix",type=str,default="wing")
args = parse.parse_args()

if __name__ == "__main__":
    # 一次运行，把全部的结果都算出来
    df_train_all = pd.read_csv(f"../data/{args.option_name}/train_data_{args.split_type}_{args.data_type}.csv")
    df_test_all = pd.read_csv(f"../data/{args.option_name}/test_data_{args.split_type}_{args.data_type}.csv")
    # all_quote_date = df_train_all["quote_date"].unique()
    all_quote_date = ["2023-01-10"]
    for date in all_quote_date:
        df_train = df_train_all[df_train_all['quote_date']==date].copy()
        df_test = df_test_all[df_test_all['quote_date']==date].copy()
        pic(df_train=df_train,
             df_test=df_test,
             option_name=args.option_name,
             data_type=args.data_type,
             date=date,
             split_type=args.split_type,
             pic_folder=args.pic_folder,
             path_prefix="20250315")
