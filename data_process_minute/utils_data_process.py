from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.interpolate import interp1d

def calculate_ttm(date, exdate):
    # 计算日期差
    ttm = (exdate - date).days / 365.25
    return ttm


def extract_forward(df):
    # 此时要保证输入的期权是同一时期记录的同一天到期的期权
    # 因此到期期限ttm相同，指数收盘价close相同
    ttm = df['ttm'][0]
    index_close = df['index_close'][0]
    forward, q, rf = np.nan, np.nan, np.nan
    atm_bound = None
    k = 3
    try:
        df_filtered = df.copy()
        if atm_bound is None:
            calls = df_filtered[df_filtered["p_close"] > df_filtered['c_close']].head(k)
            puts = df_filtered[df_filtered['p_close'] <= df_filtered['c_close']].head(k)
            atm = pd.concat((calls, puts))
        else:
            # 保持一样的数据处理方式,这里也使用7.5%
            moneyness = df_filtered['strike_price'] / index_close
            atm = df_filtered.loc[(1 - atm_bound <= moneyness) & (moneyness <= 1 + atm_bound)]
        # print(atm)
        df_filtered = atm
        df_filtered['y'] = (df_filtered['c_close'] - df_filtered['p_close'])
        df_filtered['x'] = df_filtered['strike_price']
        """model = sm.OLS(df_filtered['y'], sm.add_constant(df_filtered['x'])).fit()
        discount_factor = model.params[1]
        rf = -np.log(model.params[1]) / ttm
        q = -np.log(-model.params[0]) / ttm
        forward = index_close * np.exp((rf - q) * ttm)"""

        y = df_filtered['y']
        x = df_filtered['x']
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        discount_factor = -m
        underlying_forward = c / discount_factor
    except:
        pass

    result = pd.DataFrame({
        'date': [df['date'][0]],
        'exdate': [df['exdate'][0]],
        # 'q': [q],
        # 'rf': [rf],
        'forward': [underlying_forward],
        'index_close': [index_close],
        'ttm': [ttm],
        'discount_factor':[discount_factor]
    })

    return result

def fill_na_forward(df):
    # 合理的情况下数据集中不应该出现空值
    df_na = df[df['q'].isna()]
    df_nona = df[~df['q'].isna()]
    
    # 插值函数
    def interpolate_values(data):
        df_tmp = df_nona[df_nona['date'] == data['date'][0]]
        q_fun = interp1d(df_tmp['ttm'], df_tmp['q'], kind='linear')
        rf_fun = interp1d(df_tmp['ttm'], df_tmp['rf'], kind='linear')
        data['q'] = q_fun(data['ttm'])
        data['rf'] = rf_fun(data['ttm'])
        data['forward'] = data['index_close'] * np.exp((data['rf'] - data['q']) * data['ttm'])
        return data
    
    # 并行处理每个缺失值的日期
    df_na = df_na.groupby('date').apply(lambda x: x.groupby('exdate').apply(interpolate_values))
    
    # 合并结果并排序
    df_filled = pd.concat([df_nona, df_na]).sort_values(by=['date', 'exdate'])
    return df_filled

# 使用ThreadPoolExecutor来并行处理每个组
def process_group(group):
    # 这里调用extract_forward函数处理每个组
    processed_group = extract_forward(group)
    # 填充缺失值
    processed_group = fill_na_forward(processed_group)
    # 删除不需要的列
    processed_group = processed_group.drop(columns=['ttm', 'index_close'])
    return processed_group
