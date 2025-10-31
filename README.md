## 项目概述

这是一个隐含波动率(IV)分钟级预测的代码库,实现了多种波动率曲面拟合模型用于期权定价和预测。项目支持沪深300和中证1000等指数期权的分钟级数据处理和预测。

## 模型类型

项目包含以下几类模型:

### 1. 深度神经网络模型
- **deep_nn_line**: 深度神经网络模型,支持 CUDA 加速，利用预训练好的glow模型
- **deep_smooth**: 平滑约束的深度学习模型,使用 SVI 先验

### 2. 理论模型
- **SVI (Stochastic Volatility Inspired)**: 随机波动率启发模型
- **SABR (Stochastic Alpha Beta Rho)**: 随机 α-β-ρ 模型

## 数据处理
项目使用三阶段数据处理管道:

1. **数据加载与合约解析**: 加载期权合约 CSV 文件并解析合约信息
2. **远期价格提取**: 使用看涨看跌平价关系提取远期价格
3. **隐含波动率计算**: 使用 Black-Scholes 模型计算 IV
4. **时间分组生成**: 将分钟时间戳分组用于训练

## 快速开始

### 训练模型

以 SVI 模型为例 :

```bash
python train_minute_forecasting.py \
    --option_name hushen300_minute \
    --split_name forecast \
    --data_type clear \
    --quote_minute "2025-08-01 09:30:00" \
    --flag_column train_flag_inter
```

### 测试模型

跨分钟预测测试 :

```bash
python test_minute_forecasting.py \
    --option_name hushen300_minute \
    --minutes "2025-08-01 09:40:00" \
    --source_minute "2025-08-01 09:30:00" \
    --split_name forecast
```

## 数据格式

输入数据需包含以下列 :
- `quote_minute`: 报价时间戳
- `ttm`: 到期时间(年)
- `logm`: 对数行权价比
- `iv`: 隐含波动率
- `expiry`: 到期日
- `strike_price`: 行权价
- `forward`: 远期价格

## 评估指标

模型使用统一的评估指标 :
- **RMSE**: 均方根误差
- **MAPE**: 平均绝对百分比误差
- **Price RMSE**: 期权价格 RMSE（部分有）
- **Price MAPE**: 期权价格 MAPE（部分有）
- **But Loss**: 蝶式套利损失（部分有）

## Notes

项目支持多种训练标记列(`train_flag_forecast`, `train_flag_inter`, `train_flag_extra`)用于不同的数据划分策略。
深度学习模型支持 CUDA、MPS 和 CPU 三种设备。
所有模型都实现了跨分钟预测功能,即使用 t 时刻的模型预测 t+10 分钟的波动率曲面。