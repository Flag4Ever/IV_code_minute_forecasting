# Wing 分钟级测试脚本说明

## 文件结构

```
wing/
├── wing.py                              # Wing模型类定义
├── test_date_mult_minute.py            # 分钟级多进程测试脚本
└── script_minute/
    ├── wing_zhongzheng1000_minute.sh   # 测试执行脚本
    └── README.md                       # 本说明文档
```

## Wing 模型特点

### 与深度学习模型的本质区别

| 特性 | deep_nn_line / deep_smooth | Wing (本模型) |
|------|---------------------------|--------------|
| **模型类型** | 深度神经网络 | **理论优化模型** |
| **训练方式** | 梯度下降 (epochs) | **scipy.optimize** |
| **设备** | GPU (CUDA/MPS) | **纯CPU** |
| **并行** | 串行训练 | **多进程并行** |
| **训练耗时** | 几十分钟到几小时 | **几秒到几分钟** |
| **参数量** | 数千到数万 | **3个核心参数 (sc,pc,cc)** |
| **到期日** | 全局模型 | **每个到期日独立拟合** |

### Wing 模型公式

Wing模型使用分段函数来描述波动率曲线，将对数行权比(moneyness)空间分为6个区域：

**波动率函数**：
```
区域1 (x < dc×(1+dsm)):          vol = 常数（左侧尾部）
区域2 (dc×(1+dsm) < x ≤ dc):     vol = 左侧平滑区域
区域3 (dc < x ≤ 0):              vol = vc + sc×x + pc×x² （左侧抛物线）
区域4 (0 < x ≤ uc):              vol = vc + sc×x + cc×x² （右侧抛物线）
区域5 (uc < x ≤ uc×(1+usm)):     vol = 右侧平滑区域
区域6 (x > uc×(1+usm)):          vol = 常数（右侧尾部）
```

其中：
- x = logm = log(K/F) (对数行权比)
- **核心参数** (通过优化得到):
  - sc: 斜率参数 (skew coefficient)
  - pc: 左侧曲率参数 (put curvature)
  - cc: 右侧曲率参数 (call curvature)
- **固定参数** (默认值):
  - vc: ATM波动率 (通过插值得到)
  - dc = -0.2: 左侧拐点
  - uc = 0.2: 右侧拐点
  - dsm = 0.5: 左侧平滑范围
  - usm = 0.5: 右侧平滑范围

**优化目标**：
- 最小化加权误差 (vega-weighted norm)
- 满足无套利约束 (butterfly条件)

### Wing vs 其他模型对比

| 特性 | Wing | SVI | SABR |
|------|------|-----|------|
| **参数数量** | 3个核心参数 | 5个参数 | 3个参数 |
| **理论基础** | 分段函数参数化 | 启发式公式 | 随机波动率 |
| **拟合灵活性** | 高（6个区域） | 中等 | 中等 |
| **计算复杂度** | 中等 | 低 | 高 |
| **尾部行为** | 可控（独立区域） | 固定形式 | 固定形式 |
| **适用场景** | 复杂微笑形状 | 快速校准 | 期权定价 |

## 与日级版本的区别

| 特性 | 日级版本 | 分钟级版本 |
|------|---------|-----------|
| 数据列 | `quote_date` | `quote_minute` |
| 时间格式 | `2024-06-07` | `2024-06-07 09:30:00` |
| 数据量 | 每日一次 | 每分钟一次 |
| 处理速度 | 快 | 略慢（数据量更大） |
| 结果精度 | 日内平均 | **高频时刻精确** |

## 使用方法

### 1. 运行测试脚本

```bash
cd /Users/dxl/Documents/北航/研究生/毕业论文/波动率/IV_code_2.0.7/wing

# 方式1：直接运行shell脚本（推荐）
bash script_minute/wing_zhongzheng1000_minute.sh

# 方式2：单独测试某个划分
python test_date_mult_minute.py \
    --option_name zhongzheng1000_minute \
    --split_name extra \
    --data_type clear \
    --result_folder ./results \
    --path_prefix wing_zhongzheng1000_minute_extra
```

### 2. 在服务器上运行（也是CPU）

```bash
cd /home/douxueli/IV/IV_code_2.0.7/wing

# 确认数据文件存在
ls ../data/zhongzheng1000_minute/zhongzheng1000_minute.csv

# 运行测试（纯CPU，无需GPU）
bash script_minute/wing_zhongzheng1000_minute.sh
```

**注意**：
- Wing是**纯CPU优化**，不使用GPU
- 自动利用**多进程并行**处理多个分钟
- 默认使用 `cpu_count()//2` 个进程

## 输出文件

### 结果文件

```
results/
├── wing_zhongzheng1000_minute_extra.txt
└── wing_zhongzheng1000_minute_inter.txt
```

**结果格式**：
```
model_name  option_name  minute              data_type  split  rmse     mape     price_rmse  price_mape  price_spread  but_loss
wing        zhongzheng1000_minute  2025-08-01 09:30:00  clear      extra  0.012345  0.023456  123.456     0.234       0.345         0.456
```

### 日志文件

```
script_minute/log/
├── extra_zhongzheng1000_minute.log
└── inter_zhongzheng1000_minute.log
```

**日志内容示例**：
```log
==========================================
测试开始时间: 2025-10-15 14:23:45
划分: extra | 数据集: zhongzheng1000_minute
==========================================

加载数据: ../data/zhongzheng1000_minute/zhongzheng1000_minute.csv
共 4 个分钟需要处理
✅ 已完成 2025-08-01 09:30:00
✅ 已完成 2025-08-01 09:35:00
✅ 已完成 2025-08-01 09:40:00
✅ 已完成 2025-08-01 09:45:00

处理完成: 4 成功, 0 失败

所有任务完成！结果已保存到: ./results/wing_zhongzheng1000_minute_extra.txt

==========================================
测试结束时间: 2025-10-15 14:26:15
==========================================
```

## 性能特点

### CPU 多进程并行

```python
# 自动使用半数CPU核心
processes=os.cpu_count()//2
```

**示例**（8核CPU）：
- 使用4个并行进程
- 每个进程处理一个分钟
- 显著提升处理速度

### 处理时间估算

单个分钟处理时间：
- **到期日数量少** (5-10个): 20-30秒
- **到期日数量多** (20-30个): 60-120秒

总体时间（4个分钟）：
- **串行处理**: 4 × 90秒 = 6分钟
- **并行处理** (4进程): 90秒-2分钟

**比深度学习快30-50倍！**

## 参数说明

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--option_name` | sp500_2022_minute | 数据集名称 |
| `--data_type` | clear | 数据类型 |
| `--split_name` | extra | 数据划分 (inter/extra) |
| `--result_folder` | ./results | 结果保存目录 |
| `--path_prefix` | wing_minute | 结果文件前缀 |

### Wing 模型参数

**核心参数（通过优化得到）**：
```python
sc: (-1e3, 1e3)  # 斜率参数，控制skew方向
pc: (-1e3, 1e3)  # 左侧曲率，控制put侧曲率
cc: (-1e3, 1e3)  # 右侧曲率，控制call侧曲率
```

**固定参数（默认值）**：
```python
dc = -0.2    # 左侧拐点位置
uc = 0.2     # 右侧拐点位置
dsm = 0.5    # 左侧平滑范围
usm = 0.5    # 右侧平滑范围
vc = interp1d(x, iv)(0)  # ATM波动率（插值得到）
```

**优化配置**：
```python
epochs = 100         # 重启次数（随机初始值）
epsilon = 1e-6       # 容差
use_constraints = True  # 使用无套利约束
```

## 与其他模型的对比

### 计算复杂度

| 模型 | 参数数量 | 单次拟合时间 | GPU需求 |
|------|---------|------------|---------|
| **Wing** | **3** | **秒级** | **❌ 无** |
| SVI | 5 | 秒级 | ❌ 无 |
| SABR | 3 | 秒级 | ❌ 无 |
| deep_nn_line | ~5000 | 小时级 | ✅ 需要 |

### 优势与劣势

**Wing优势**：
- ✅ **灵活性高** - 6个区域独立控制
- ✅ **尾部可控** - 常数尾部避免极端值
- ✅ **无GPU依赖** - 纯CPU运行
- ✅ **平滑过渡** - 平滑区域确保连续性
- ✅ **适应性强** - 适合复杂微笑形状

**Wing劣势**：
- ❌ **参数较多** - 固定参数需要调整
- ❌ **优化复杂** - 比SVI稍慢
- ❌ **每个到期日独立** - 无法利用到期日间关系
- ❌ **理论性弱** - 缺乏深层理论基础

### 使用建议

**使用Wing的场景**：
- ✅ 复杂的波动率微笑形状
- ✅ 需要控制尾部行为
- ✅ 深度虚值/实值期权重要
- ✅ 多段拟合需求

**使用SVI的场景**：
- ✅ 快速实时定价
- ✅ 简单微笑形状
- ✅ 参数直观性重要
- ✅ 计算效率优先

**使用SABR的场景**：
- ✅ 需要理论支撑
- ✅ 期权定价和风险管理
- ✅ 行业标准应用
- ✅ 高精度拟合

**使用深度学习的场景**：
- ✅ 追求最高精度
- ✅ 复杂市场模式
- ✅ 有充足GPU资源
- ✅ 可接受训练时间

## 常见问题

### Q1: Wing需要GPU吗？

**答**：不需要。Wing使用scipy.optimize进行数值优化，完全基于CPU。

### Q2: Wing模型的6个区域是什么？

**答**：
1. **区域1**：左侧常数尾部 (x < dc×(1+dsm))
2. **区域2**：左侧平滑区域 (dc×(1+dsm) < x ≤ dc)
3. **区域3**：左侧抛物线 (dc < x ≤ 0) - put侧
4. **区域4**：右侧抛物线 (0 < x ≤ uc) - call侧
5. **区域5**：右侧平滑区域 (uc < x ≤ uc×(1+usm))
6. **区域6**：右侧常数尾部 (x > uc×(1+usm))

### Q3: 为什么Wing有常数尾部？

**答**：
- 避免深度虚值期权波动率趋向无穷或负值
- 更符合市场实际观察（尾部波动率相对稳定）
- 提高数值稳定性

### Q4: 如何调整固定参数？

```python
# 修改 test_date_mult_minute.py 中的固定参数：
dc, uc, dsm, usm = -0.3, 0.3, 0.6, 0.6  # 扩大拐点和平滑范围

# 常见调整场景：
# 1. 扩大范围：dc=-0.3, uc=0.3 (更宽的拟合区域)
# 2. 增加平滑：dsm=0.8, usm=0.8 (更平滑的过渡)
# 3. 对称设置：dc=-uc, dsm=usm (对称微笑)
```

### Q5: 如何提高速度？

```bash
# 方法1：增加进程数（如果CPU核心充足）
# 修改 test_date_mult_minute.py 中的：
processes=os.cpu_count()  # 使用所有核心

# 方法2：减少优化epochs
# 修改 test_date_mult_minute.py 中的：
epochs=50  # 从100减少到50

# 方法3：放宽容差
epsilon=1e-5  # 从1e-6放宽到1e-5
```

### Q6: 结果如何解读？

```
rmse=0.012345  # 隐含波动率均方根误差（越小越好）
mape=0.023456  # 平均百分比误差（越小越好）
but_loss=0.456 # 蝶式套利损失（越小越好，理想为0）
```

**参考标准**：
- RMSE < 0.01: 优秀
- RMSE < 0.02: 良好
- RMSE < 0.05: 可接受
- RMSE > 0.05: 较差

### Q7: 出现优化失败怎么办？

**常见原因**：
1. 数据质量问题（异常值、缺失值）
2. 初始参数不合适
3. 约束条件过严

**解决方法**：
```python
# 1. 关闭约束
use_constraints=False  # 先不使用无套利约束

# 2. 增加epochs
epochs=200  # 增加重启次数

# 3. 放宽参数范围
is_bound_limit=False  # 不限制参数范围

# 4. 检查数据
print(df_train[['logm', 'iv', 'expiry']].describe())
```

## 高级用法

### 1. 自定义固定参数

修改 `test_date_mult_minute.py` 中的固定参数：
```python
# 默认对称设置
dc, uc, dsm, usm = -0.2, 0.2, 0.5, 0.5

# 非对称设置（put侧更陡峭）
dc, uc, dsm, usm = -0.3, 0.2, 0.3, 0.5

# 宽松设置（更大范围）
dc, uc, dsm, usm = -0.4, 0.4, 0.8, 0.8
```

### 2. 修改优化策略

```python
# 在 wing.py 的 calibrate() 方法中修改：

# 更快速的优化（减少epochs）
epochs=50

# 更精确的优化（增加epochs，降低epsilon）
epochs=200
epsilon=1e-8

# 使用不同优化方法
method="trust-constr"  # 或 "COBYLA"
```

### 3. 并行处理配置

```python
# 修改 test_date_mult_minute.py
# 使用所有CPU核心
processes=os.cpu_count()

# 或手动指定
processes=16  # 使用16个进程
```

### 4. Vega加权

```python
# 当前使用均匀权重
vega = np.ones_like(train_x)

# 可改为实际vega权重（如果有数据）
vega = data_use["vega"].values  # 使用真实vega
```

## 验证清单

运行前检查：
- [ ] 数据文件 `zhongzheng1000_minute.csv` 存在
- [ ] 数据包含必要列：`quote_minute`, `ttm`, `logm`, `iv`, `expiry`
- [ ] results 目录可写
- [ ] 有足够的磁盘空间（结果文件通常几MB）

运行后检查：
- [ ] 结果文件生成在 `results/` 目录
- [ ] 日志文件显示"处理完成"
- [ ] 所有分钟都有结果（无错误）
- [ ] RMSE和MAPE在合理范围内
- [ ] 没有大量优化失败警告

## 故障排查

### 问题1：大量 "calibrate wing-model wrong" 警告

**原因**：优化未收敛

**解决**：
```python
# 1. 增加epochs
epochs=200  # 更多重启次数

# 2. 放宽容差
epsilon=1e-5  # 降低精度要求

# 3. 关闭约束试试
use_constraints=False
```

### 问题2：进程卡住不动

**原因**：某个分钟的优化陷入死循环

**解决**：
```bash
# 查看哪个进程在运行
ps aux | grep test_date_mult_minute

# 杀掉卡住的进程
pkill -f test_date_mult_minute

# 增加超时时间
msg_type, content = args_queue.get(timeout=6000)  # 100分钟
```

### 问题3：内存不足

**原因**：并行进程过多

**解决**：
```python
# 减少进程数
processes=os.cpu_count()//4  # 只用1/4的核心
```

### 问题4：结果文件为空

**原因**：所有优化都失败

**检查**：
```bash
# 查看日志中的错误信息
cat script_minute/log/extra_zhongzheng1000_minute.log

# 单独测试一个分钟
python test_date_mult_minute.py \
    --option_name zhongzheng1000_minute \
    --split_name extra
```

## 理论背景

### Wing模型起源

Wing模型由Benaim, Dodgson和Kainth于2008年提出，是一种灵活的波动率曲线参数化方法。

**设计理念**：
- 分段函数设计，每个区域有明确用途
- 中心区域使用抛物线（二次函数）
- 平滑区域确保连续性
- 常数尾部防止极端值

### 无套利条件

模型通过约束确保满足：
1. **Butterfly条件**：二阶导数非负
2. **Call spread条件**：一阶导数有界

**验证方法**：
对每个区域分别检查蝶式套利条件，确保：
```
d²C/dK² ≥ 0  (butterfly)
```

### 参数经济含义

- **sc (skew coefficient)**: 控制微笑的斜率/偏度
- **pc (put curvature)**: 控制put侧的曲率
- **cc (call curvature)**: 控制call侧的曲率
- **vc (ATM vol)**: 平价波动率水平
- **dc, uc**: 拐点位置，定义抛物线区域范围
- **dsm, usm**: 平滑范围，控制过渡区域宽度

### 与市场微笑的对应

| 市场特征 | Wing参数 |
|---------|---------|
| 整体波动率水平 | vc (ATM波动率) |
| Skew方向 | sc (斜率参数) |
| Put侧陡峭程度 | pc (左侧曲率) |
| Call侧陡峭程度 | cc (右侧曲率) |
| 微笑宽度 | dc, uc (拐点位置) |
| 尾部平坦度 | dsm, usm (平滑范围) |

## 参考文献

1. Benaim, S., Dodgson, M., & Kainth, D. (2008). "An arbitrage-free method for smile construction."
2. Wystup, U. (2010). "FX Options and Structured Products." Chapter on volatility smile modeling.
3. Kahale, N. (2004). "An arbitrage-free interpolation of volatilities." Risk Magazine, 17(5), 102-106.
