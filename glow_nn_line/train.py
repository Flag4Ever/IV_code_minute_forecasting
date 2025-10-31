from tqdm import tqdm
import numpy as np

import argparse
import os

import torch
from torch import optim
from torch.utils.data import DataLoader

from gen_model import Glow
import pandas as pd
import data_utils

parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--data_channel",default=1,type=int,help="data channel")
parser.add_argument("--batch", default=16, type=int, help="batch size")
parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument(
    "--n_flow", default=32, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument("--filter_size", default=128, type=int, help="size of filter")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--w_d",default=1e-4, type=float, help= "weight decay")
parser.add_argument("--line_size", default=64, type=int, help="line size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
parser.add_argument("--noise_scale",default=0.02, type=float, help="the std of noise")
parser.add_argument("--data_path",default="../data/generated_data/ssvi_train.csv",type=str, help="train data")
parser.add_argument("--checkpoint_path",default="./checkpoint",type=str,help="checkpoint path")
parser.add_argument("--sample_path",default="./sample",type=str,help="sample data path")
parser.add_argument("--device",default="cuda:0",type=str,help="train device")
args = parser.parse_args()
# parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")

def sample_data(data_path,length_num,channel_num=1):
    # data_path = "./data/ssvi_train.csv"
    data = pd.read_csv(data_path,index_col=None,header=None,dtype=np.float32).values
    data = torch.tensor(data, dtype=torch.float32)
    # dataset = TensorDataset(data)
    surface_data = data_utils.SurfaceData(data,length=length_num,channel=channel_num)
    loader = DataLoader(surface_data, batch_size=args.batch, shuffle=True,num_workers=24)
    # loader = DataLoader(surface_data, batch_size=args.batch, shuffle=True)
    loader = iter(loader)
    while True:
        try:
            yield next(loader)
        except StopIteration:
            loader = DataLoader(surface_data, batch_size=args.batch, shuffle=True)
            loader = iter(loader)
            yield next(loader)


def calc_z_shapes(n_channel, length_size, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        length_size //= 2
        z_shapes.append((n_channel, length_size))

    length_size //= 2
    z_shapes.append((n_channel * 2, length_size))

    return z_shapes

def calc_loss(log_p,logdet,line_size,data_channel):
    # 新损失函数
    n_pixel = line_size * data_channel
    loss = logdet+log_p
    return (
        ((-loss/n_pixel).mean()),
        ((log_p/n_pixel).mean()),
        ((logdet/n_pixel).mean()),
    )


def train(args, model, optimizer, data_path):
    # 看一下原始的实验是如何进行归一化的
    model.train()
    compile_model = torch.compile(model,mode="max-autotune")
    dataset = iter(sample_data(data_path,length_num=args.line_size,channel_num=args.data_channel))
    z_sample = []
    z_shapes = calc_z_shapes(args.data_channel, args.line_size, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))
    
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)


    with tqdm(range(args.iter)) as pbar:
    # with range(args.iter) as pbar:
        for i in pbar:
    # for i in range(args.iter):
            # 相当于说是减少了图像的深度
            image = next(dataset)
            image = image.to(device)
            # image = data_utils.min_max_normalize(image)
            image, log_det = data_utils.logit_transform(image)
            # compile在第一次前向时要做图捕捉和Kernel调度，成本较高，所以让首个batch只跑一遍forward触发编译，不保存梯度和计算损失
            if i == 0:
                with torch.no_grad():
                    # log_p, logdet, _ = model.module(
                    log_p, logdet, z_out = compile_model(
                        image
                    )
                    continue
            else:
                log_p, logdet, _ = compile_model(image)
            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.line_size,args.data_channel)
            compile_model.zero_grad()
            loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            pbar.set_description(
                f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            )
            """if (i+1) % 1000 == 0:
                print(f"Epoch: {i+1} Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}")         
            """
            if (i+1) % 5000 == 0:
                with torch.no_grad():
                    # 这里的保存方式是需要修改一下
                    # 这里显然是调用了全局变量model_single
                    # iv_surface = model_single.reverse(z_sample).cpu()
                    iv_surface = compile_model.reverse(z_sample).cpu()
                    iv_surface, _ = data_utils.logit_transform(iv_surface, reverse=True)
                    iv_surface = torch.mean(iv_surface, dim=1, keepdim=True)
                    # print(iv_surface.shape)
                    iv_surface = iv_surface.data.reshape(args.n_sample,args.line_size)
                    np.savetxt(f"{args.sample_path}/ssvi_{i+1}.csv",iv_surface,fmt="%.6f",delimiter=",")

            if (i+1) % 50000 == 0:
                torch.save(
                    model.state_dict(), f"{args.checkpoint_path}/model_{i+1}.pt"
                )

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.set_float32_matmul_precision('high')
    device = torch.device(args.device if torch.backends.mps.is_available() else "cpu")
    print(args)

    model = Glow(
        args.data_channel, args.n_flow, args.n_block, args.line_size, filter_size=args.filter_size,affine=args.affine, conv_lu=not args.no_lu
    )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.w_d)

    train(args, model, optimizer,data_path=args.data_path)