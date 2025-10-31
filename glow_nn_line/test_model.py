from tqdm import tqdm
import numpy as np
from pprint import pprint
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

def test(args, model, optimizer, data_path):
    gen_df = pd.read_csv("ssvi.csv",index_col=None,header=None,dtype=np.float32)
    gen_df = gen_df.values.reshape(-1,1,128)
    test_data = torch.tensor(gen_df,device=args.device)
    print(test_data.shape)
    # print(test_data)
    test_data,_ = data_utils.logit_transform(test_data)
    model.load_state_dict(torch.load(f"{args.checkpoint_path}/model_500000.pt",weights_only=True))
    pprint(model.state_dict())
    # compile_model = torch.compile(model,mode="max-autotune")
    compile_model = model
    pprint(compile_model.state_dict())
    print(test_data)
    log_p2,logdet2,z_out = compile_model(test_data)
    logdet2 = logdet2.mean()
    loss2, log_p2, log_det2 = calc_loss(log_p2, logdet2, args.line_size,args.data_channel)
    print(f"Valid: Loss: {loss2.item():.5f}; logP: {log_p2.item():.5f}; logdet: {log_det2.item():.5f}")

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    # torch.set_float32_matmul_precision('high')
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

    model = Glow(
        args.data_channel, args.n_flow, args.n_block, args.line_size, filter_size=args.filter_size,affine=args.affine, conv_lu=not args.no_lu
    )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.w_d)

    test(args, model, optimizer,data_path=args.data_path)
