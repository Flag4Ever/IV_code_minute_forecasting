# 加载隐含波动率曲面数据
import torch
import torch.nn.functional as F
import torch.distributions as distributions
import torch.utils.data as data

import numpy as np
import pandas as pd

class SurfaceData(data.Dataset):
    # 将其中一个维度设置为1
    # [b,1,128]
    def __init__(self, data,length=128,channel=1):
        # data输入为N个28行x28列
        single_data = data.reshape(-1,1,length)
        self.data = np.repeat(single_data, channel, axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return x

def min_max_normalize(data):
    # 计算数据的最小值和最大值
    min_val = torch.min(data, dim=-1, keepdim=True).values  # 计算最小值
    max_val = torch.max(data, dim=-1, keepdim=True).values  # 计算最大值
    
    # 避免除以零的情况
    if torch.any(max_val - min_val == 0):
        raise ValueError("数据的最大值和最小值相同，无法进行Min-Max标准化。")
    
    # 进行Min-Max标准化
    normalized_data = (data - min_val) / (max_val - min_val)
    
    return normalized_data

def logit_transform(x, constraint=0.9, reverse=False):
    '''Transforms data from [0, 1] into unbounded space.
    如果是值在[0,2]之间的波动率数据，则不需要乘以2

    Restricts data into [0.05, 0.95].
    Calculates logit(alpha+(1-alpha)*x).

    Args:
        x: input tensor.
        constraint: data constraint before logit.
        reverse: True if transform data back to [0, 1].
    Returns:
        transformed tensor and log-determinant of Jacobian from the transform.
        (if reverse=True, no log-determinant is returned.)
    '''
    if reverse:
        x = 1. / (torch.exp(-x) + 1.)    # [0.05, 0.95]
        x *= 2.             # [0.1, 1.9]
        x -= 1.             # [-0.9, 0.9]
        x /= constraint     # [-1, 1]
        x += 1.             # [0, 2]
        # x /= 2.             # [0, 1] 波动率不需要再除2
        return x, 0
    else:
        [B, C, L] = list(x.size())
        noise = torch.rand(B, C, L, device=x.device)
        x = (x * 255 + noise) / 256
        # restrict data
        # x *= 2.             # [0, 2] 波动率不需要再乘2
        x -= 1.             # [-1, 1]
        x *= constraint     # [-0.9, 0.9]
        x += 1.             # [0.1, 1.9]
        x /= 2.             # [0.05, 0.95]

        # logit data
        logit_x = torch.log(x) - torch.log(1. - x)

        # log-determinant of Jacobian from the transform
        pre_logit_scale = torch.tensor(
            np.log(constraint) - np.log(1. - constraint))
        log_diag_J = F.softplus(logit_x) + F.softplus(-logit_x) \
            - F.softplus(-pre_logit_scale)

        return logit_x, torch.sum(log_diag_J, dim=(1, 2))
