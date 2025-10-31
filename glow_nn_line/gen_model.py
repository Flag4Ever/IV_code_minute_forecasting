# 算子融合
# 确保乘法效率
# 减少CPU-GPU交互
# 启用2.0编译
# 内存分配策略优化

import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi

def logabs(x):
    return torch.log(torch.abs(x) + 1e-6)  # 添加小偏移避免log(0)

# 第一部分
class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            # 直接在通道和长度维度计算均值和方差
            mean = input.mean(dim=(0, 2), keepdim=True)  # shape [1, in_channel, 1]
            std = input.std(dim=(0, 2), keepdim=True, unbiased=False) + 1e-6
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / std)

    def forward(self, input):
        _, _, length = input.shape
        log_abs = logabs(self.scale)
        logdet = length * torch.sum(log_abs)
        if self.logdet:
            return self.scale * (input + self.loc), logdet
        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc

# 第二个部分
class InvConv1d(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        weight = torch.randn(in_channel,in_channel)
        q,_ = torch.linalg.qr(weight,"reduced")
        self.weight = nn.Parameter(q.unsqueeze(2))

    def forward(self, input):
        _,_,length = input.shape
        out = F.conv1d(input,self.weight)
        logdet = (
            length * torch.slogdet(self.weight.squeeze().double())[1].float()
        )
        return out, logdet
    
    def reverse(self, output):
        return F.conv1d(
            output,torch.linalg.inv(self.weight.squeeze()).unsqueeze(2)
        )

# 第二部分：采用PyTorch实现的LU分解
class InvConv1dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        weight = torch.randn(in_channel, in_channel, dtype=torch.float32)
        q, _ = torch.linalg.qr(weight)
        lu, pivots = torch.linalg.lu_factor(q)
        w_p, w_l, w_u = torch.lu_unpack(lu, pivots)
        
        w_s = torch.diag(w_u)
        w_u = torch.triu(w_u, diagonal=1)
        u_mask = torch.triu(torch.ones_like(w_u), diagonal=1)
        l_mask = u_mask.T
        
        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", u_mask)
        self.register_buffer("l_mask", l_mask)
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0], dtype=torch.float32))
        
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, length = input.shape
        weight = self.calc_weight()
        out = F.conv1d(input, weight)
        logdet = length * torch.sum(self.w_s)
        return out, logdet

    def calc_weight(self):
        # 更稳定的权重计算
        l_matrix = self.w_l * self.l_mask + self.l_eye
        u_matrix = (self.w_u * self.u_mask) + torch.diag(self.s_sign * (torch.exp(self.w_s) + 1e-6))
        weight = self.w_p @ l_matrix @ u_matrix
        return weight.unsqueeze(2)
    def reverse(self, output):
        weight = self.calc_weight().squeeze(2)
        # 使用更稳定的求解方式代替直接求逆
        weight_inv = torch.linalg.solve(weight, torch.eye(weight.shape[0], device=weight.device))
        return F.conv1d(output, weight_inv.unsqueeze(2))

    
class ZeroConv1d(nn.Module):
    # 将卷积参数设置为0,保证最开始的时候是一个恒等变换
    def __init__(self, in_channel, out_channel, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, 3, padding=padding)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1))

    
    def forward(self, input):
        out = F.pad(input, [1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)
        return out

# 第三个部分
class ConVNN(nn.Module):
    # 新增网络结构，在卷积层后加一个全连接层
    # 输入形状[B,in_channel,length]
    def __init__(self,in_channel,filter_size,kernel_size,length,affine=True):
        super().__init__()
        self.length = length
        self.filter_size = filter_size
        self.affine = affine
        self.conv1d = nn.Conv1d(in_channel,filter_size,kernel_size,padding="same")
        # 输出形状[B,filter_size,length]
        self.conv1d.weight.data.normal_(0, 0.05)
        self.conv1d.bias.data.zero_()
        self.relu = nn.ReLU(inplace=True)
        # 期望输出[B,in_channel,length]
        self.mlp = nn.Linear(filter_size*length,in_channel*2*length if self.affine else in_channel*length)
        nn.init.zeros_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)
    
    def forward(self,input):
        B,C,L = input.shape
        y = self.conv1d(input)
        # print(y.shape)
        y = self.relu(y)
        # print(f"y shape {y.shape}")
        # print(f"length {self.length}")
        # print(f"filter_size {self.filter_size}")
        y = y.reshape(B,self.filter_size*self.length)
        y = self.mlp(y)
        if self.affine:
            y = y.reshape(B,C*2,L)
        else:
            y = y.reshape(B,C,L)
        return y

# 第四个部分
class AffineCoupling(nn.Module):
    # 适合1D卷积的仿射耦合层 保留length参数
    def __init__(self, in_channel, length,filter_size=256, affine=True):
        super().__init__()
        self.affine = affine
        """self.net = nn.Sequential(
            nn.Conv1d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv1d(filter_size, in_channel if self.affine else in_channel // 2),
        )
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()"""
        self.net = ConVNN(in_channel=in_channel//2,
                          filter_size=filter_size,
                          kernel_size=3,
                          length=length,
                          affine=affine)
    
    def forward(self, input):
        # 处理输入
        # 首先按照第1维进行切分为两部分
        in_a, in_b = input.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            s = torch.sigmoid(log_s + 2)
            out_b = (in_b + t) * s
            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)
        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None
        return torch.cat([in_a, out_b], 1), logdet
    def reverse(self, output):
        # 反向过程
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = torch.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t
        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out
        return torch.cat([out_a, in_b], 1)

# 各个部分组装成一个flow
class Flow(nn.Module):
    # 网络结构的一块流
    # 这一部分并没有涉及到维度切断内容
    def __init__(self, in_channel,length,filter_size, affine=True, conv_lu=True):
        super().__init__()
        # 模型包含的网络结构
        # 第一部分是标准化层
        self.actnorm = ActNorm(in_channel)
        # 第二部分是可逆卷积层
        if conv_lu:
            self.invconv = InvConv1dLU(in_channel)
        else:
            self.invconv = InvConv1d(in_channel)
        # 第三部分是仿射耦合层
        self.coupling = AffineCoupling(in_channel, length=length, filter_size=filter_size, affine=affine)
    
    def forward(self, input):
        # 前向过程：
        # 先经过标准化层,再经过可逆卷积层
        # 最后经过耦合层
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)
        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input

# 计算高斯分布的概率
def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

# 从高斯分布中采样
def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps

# 组装成一个block
class Block(nn.Module):
    # 相当于每一个Block才会有一次维度裁剪
    def __init__(self, in_channel, n_flow, length,filter_size, split=True, affine=True, conv_lu=True):
        super().__init__()
        squeeze_dim = in_channel * 2
        length //= 2
        self.flows = nn.ModuleList()
        for i in range(n_flow):
            # 每一块中的flow是统一大小的
            self.flows.append(Flow(squeeze_dim,length=length, filter_size=filter_size, affine=affine, conv_lu=conv_lu))
        # 决定这一个块是否切分
        self.split = split
        if split:
            self.prior = ZeroConv1d(in_channel, in_channel * 2)
        else:
            self.prior = ZeroConv1d(in_channel * 2, in_channel * 4)

    
    def forward(self, input):
        b_size, n_channel, length = input.shape
        # 将输入形状进行转换,多出一个维数是2的维度
        squeezed = input.view(b_size, n_channel, length // 2, 2)
        # b_size,n_channel,2,1,height//2,width
        squeezed = squeezed.permute(0, 1, 3, 2)
        out = squeezed.contiguous().view(b_size, n_channel * 2, length // 2)

        logdet = 0
        # 将squeezed后的数据过flow层
        num_flow = 0
        for flow in self.flows:
            num_flow += 1
            # print(f"----------{num_flow}---------")
            out, det = flow(out)
            logdet = logdet + det
            # print(out.shape)

        if self.split:
            # 先对输出进行切分
            # 切分出的一部分直接作为输出
            # print(out.shape)
            out, z_new = out.chunk(2, 1)
            # 另一部分作为均值和方差输出
            # 初始置为标准正态分布
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            # 不切分，直接利用零向量计算均值和方差
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out
        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):
        # 反向过程
        # 先把第一层进行逆转
        input = output
        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)
            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)
            else:
                zero = torch.zeros_like(input)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z
        # 经过逆转后，放入flow模型中进行反向过程
        for flow in self.flows[::-1]:
            input = flow.reverse(input)
        # 将squeezed后的张量进行unsqueezed
        b_size, n_channel, lenght = input.shape
        unsqueezed = input.view(b_size, n_channel // 2, 2, lenght)
        unsqueezed = unsqueezed.permute(0, 1, 3, 2)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 2, lenght * 2
        )
        return unsqueezed

class Glow(nn.Module):
    def __init__(
        self, in_channel, n_flow, n_block, length,filter_size, affine=True, conv_lu=True
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            # 根据块数的不同,对通道进行修改
            self.blocks.append(Block(n_channel, n_flow, length=length,
                                     filter_size=filter_size, affine=affine, conv_lu=conv_lu))
            length //= 2
        self.blocks.append(Block(n_channel, n_flow, length=length,
                                 filter_size=filter_size, split=False, affine=affine))

    def forward(self, input):
        # 前向过程
        # 输出是三部分
        # 第一部分是概率
        # 第二部分是det值
        # 第三部分是模型转换之后的z_outs
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p
        return log_p_sum.unsqueeze(0), logdet.unsqueeze(0), z_outs

    def reverse(self, z_list, reconstruct=False):
        # 模型逆向输出结果
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)
            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)
        return input
