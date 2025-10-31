import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.prior_model import PriorModel
# 模型没有经过优化,速度较慢
def calc_loss(log_p, logdet, image_size):
    n_pixel = image_size * image_size
    loss = logdet + log_p
    return (
        (-loss / (np.log(2) * n_pixel)).mean(),
        (log_p / (np.log(2) * n_pixel)).mean(),
        (logdet / (np.log(2) * n_pixel)).mean(),
    )

class IvSmoother(nn.Module):
    # 其实可以用父类将代码进行进一步优化
    def __init__(self, neurons_vec, activation='relu', penalty = dict(fit = 1, c4 = 10, c5 = 10, c6 = 10, atm = 0.1),prior=None, phi_fun=None, w_atm_fun=None, spread=False):
        super(IvSmoother, self).__init__()
        # 构建神经网络模型
        self.neurons_vec = neurons_vec
        self.activation = activation
        self.penalty = penalty
        self.prior = prior
        self.phi_fun = phi_fun
        self.spread = spread
        # Define the activation function
        if activation == 'relu':
            self.afun = F.relu
        elif activation == 'softplus':
            self.afun = F.softplus
        elif activation == "sigmoid":
            self.afun = torch.sigmoid
        elif activation == "tanh":
            self.afun = torch.tanh
        else:
            raise ValueError("Unsupported activation function")
        # Define layers
        self.layers = nn.ModuleList()
        # self.layer_norm = nn.LayerNorm(neurons_vec[-1])
        n_input = 2
        for i, n_neurons in enumerate(neurons_vec):
            layer = nn.Linear(n_input, n_neurons)
            nn.init.trunc_normal_(layer.weight, mean=0, std=math.sqrt(1.0 / (n_input + n_neurons)),a=-0.25,b=0.25)
            nn.init.trunc_normal_(layer.bias, mean=0, std=math.sqrt(1.0 / (n_input + n_neurons)),a=-0.25,b=0.25)
            self.layers.append(layer)
            n_input = n_neurons
        self.output_layer = nn.Linear(n_input, 1)
        # self.softplus = nn.Softplus()
        self.alpha = nn.Parameter(torch.tensor([torch.log(torch.exp(torch.tensor(1.0)) - 1)]))
        # 加入先验模型
        if self.prior is not None:
            self.prior_model = PriorModel(prior)
        self.w_atm_fun = w_atm_fun
    def get_ann_out(self,ttm,logm):
        x = torch.stack((ttm, logm), dim=-1)
        for i, layer in enumerate(self.layers):
            x = self.afun(layer(x))
        # x = self.layer_norm(x)
        y = self.output_layer(x)
        ann_out = torch.exp(self.alpha)*(1+torch.tanh(y))
        return ann_out.flatten()

    def forward(self, ttm, logm):
        w_nn = self.get_ann_out(ttm,logm)
        if self.prior is not None:
            w_atm = torch.from_numpy(np.clip(self.w_atm_fun(ttm.detach().cpu().numpy()),a_min=1e-6,a_max=2)).to(torch.float32)
            w_atm = w_atm.to(ttm.device)
            w_prior = self.prior_model(logm, w_atm)
            w_hat = w_nn*w_prior
        else:
            w_hat = w_nn
        return w_hat
    
    def compute_w_hat(self, ttm, logm):
        return self.forward(ttm, logm)
    
    def compute_iv_hat(self, ttm, logm):
        w_hat = self.compute_w_hat(ttm, logm)
        iv_hat = torch.sqrt(w_hat / (ttm+1e-6))
        return iv_hat
    def get_loss_fit_iv(self, iv, iv_hat, iv_spread=None):
        l_fit_iv_mape = torch.mean(torch.abs((iv_hat - iv) / (iv + 1e-6)))

        if iv_spread is None:
            # l_fit_iv_rmse = torch.mean((1e-6 + (iv - iv_hat) ** 2) ** 0.5)
            l_fit_iv_rmse = torch.sqrt(torch.mean(((iv - iv_hat) ** 2)))
        else:
            l_fit_iv_rmse = torch.mean(1e-6 + torch.abs(iv - iv_hat) / (1 + iv_spread))
        
        l_fit_iv = (l_fit_iv_rmse + l_fit_iv_mape)
        return {
            "l_fit_iv_rmse": l_fit_iv_rmse,
            "l_fit_iv_mape": l_fit_iv_mape,
            "l_fit_iv": l_fit_iv
        }
    
    def get_loss_fit_w(self, w, w_hat):
        l_fit_w_rmse = torch.sqrt(torch.mean(((w-w_hat)**2)))
        l_fit_w_mape = torch.mean(torch.abs((w_hat - w) / (w + 1e-6)))
        l_fit_w = (l_fit_w_rmse + l_fit_w_mape)
        return {
            "l_fit_w_rmse": l_fit_w_rmse,
            "l_fit_w_mape": l_fit_w_mape,
            "l_fit_w": l_fit_w
        }

    def get_loss_fit(self, w, w_hat, iv, iv_hat, iv_spread=None,prior=None):
        # l_fit_w_rmse = torch.mean((1e-6 + (w - w_hat) ** 2) ** 0.5)
        l_fit_w_rmse = torch.sqrt(torch.mean(((w-w_hat)**2)))
        l_fit_w_mape = torch.mean(torch.abs((w_hat - w) / (w + 1e-6)))
        l_fit_w = (l_fit_w_rmse + l_fit_w_mape)
        # print(torch.abs((iv_hat-iv)/iv))
        l_fit_iv_mape = torch.mean(torch.abs((iv_hat - iv) / (iv + 1e-6)))

        if iv_spread is None:
            # l_fit_iv_rmse = torch.mean((1e-6 + (iv - iv_hat) ** 2) ** 0.5)
            l_fit_iv_rmse = torch.sqrt(torch.mean((iv - iv_hat) ** 2))
        else:
            l_fit_iv_rmse = torch.mean(torch.abs(iv - iv_hat) / (1 + iv_spread))
        
        l_fit_iv = (l_fit_iv_rmse + l_fit_iv_mape)

        return {
            "l_fit_w_rmse": l_fit_w_rmse,
            "l_fit_w_mape": l_fit_w_mape,
            "l_fit_w": l_fit_w,
            "l_fit_iv_rmse": l_fit_iv_rmse,
            "l_fit_iv_mape": l_fit_iv_mape,
            "l_fit_iv": l_fit_iv
        }
    
    def get_loss_arb(self, ttm, logm, c6_num):
        # 将c4c5c6组合到一块儿
        # 只算一次
        # 计算的时候再挑出来
        inputs = torch.stack([ttm, logm], dim=1).requires_grad_(True)
        w = self.compute_w_hat(inputs[:, 0], inputs[:, 1])
        dw = torch.autograd.grad(w, inputs, torch.ones_like(w), 
                               create_graph=True, retain_graph=True)[0]
        dwdt, dwdk = dw[:, 0], dw[:, 1]
        # 计算二阶导数
        d2wdk2 = torch.autograd.grad(dwdk, inputs, torch.ones_like(dwdk),
                                   create_graph=True, retain_graph=True)[0][:, 1]
        # 预计算公共项
        w_eps = w + 1e-6
        term1 = (1 - (logm * dwdk) / (2 * w_eps)) ** 2
        term2 = (dwdk ** 2) * (1 / (4 * w_eps) + 0.0625)
        term3 = d2wdk2 / 2
        return {
            "l_c4": F.relu(-dwdt).mean(),
            "l_c5": F.relu(-(term1 - term2 + term3)).mean(),
            "l_c6": d2wdk2[-c6_num:].abs().mean(),
        }
    
    def get_train_loss(self,data_dict,pred_type="iv"):
        iv_hat_fit = self.compute_iv_hat(data_dict["ttm_fit:0"],data_dict["logm_fit:0"])
        w_hat_fit = iv_hat_fit**2*(data_dict["ttm_fit:0"]+1e-6)
        w_fit = data_dict["w:0"]
        iv_fit = data_dict["iv:0"]
        if pred_type == "iv":
            loss_fit = self.get_loss_fit_iv(iv_fit,iv_hat_fit)["l_fit_iv"]
        elif pred_type == "w":
            loss_fit = self.get_loss_fit_w(w_fit,w_hat_fit)["l_fit_w"]
        
        c6_num = len(data_dict["ttm_c6:0"])
        ttm_c4c5c6 = torch.cat((data_dict["ttm_c4c5:0"],data_dict["ttm_c6:0"]),dim=0)
        logm_c4c5c6 = torch.cat((data_dict["logm_c4c5:0"],data_dict["logm_c6:0"]),dim=0)
        arb_loss = self.get_loss_arb(ttm=ttm_c4c5c6,logm=logm_c4c5c6,c6_num=c6_num)
        logm_atm = torch.zeros_like(data_dict["ttm_c4c5:0"])
        ann_output = self.get_ann_out(data_dict["ttm_c4c5:0"],logm_atm)
        loss_atm = self.get_loss_atm(ann_output)["l_atm"]
        return self.penalty["fit"]*loss_fit+self.penalty["c4"]*arb_loss["l_c4"]+self.penalty["c5"]*arb_loss["l_c5"]+self.penalty["c6"]*arb_loss["l_c6"]+self.penalty["atm"]*loss_atm


    def get_loss_atm(self, ann_output):
        if self.prior is None:
            l_atm = torch.tensor(0.0)
        elif self.prior in ["svi", "bs"]:
            # l_atm = torch.mean((1e-6 + (ann_output - 1.0) ** 2) ** 0.5)
            l_atm = torch.sqrt(torch.mean((1e-6 + (ann_output - 1.0)**2)))
        else:
            raise ValueError("Unknown prior")
        return {"l_atm": l_atm}
    
    """def get_test_loss(self,data_dict):
        # 默认是iv的相关指标
        iv_hat_fit = self.compute_iv_hat(data_dict["ttm_fit:0"],data_dict["logm_fit:0"])
        w_hat_fit = self.compute_w_hat(data_dict["ttm_fit:0"],data_dict["logm_fit:0"])
        w_fit = data_dict["w:0"]
        iv_fit = data_dict["iv:0"]
        loss_fit = self.get_loss_fit(w_fit,w_hat_fit,iv_fit,iv_hat_fit)
        w_hat_c4c5 = self.compute_w_hat(data_dict["ttm_c4c5:0"],data_dict["logm_c4c5:0"])
        c4c5 = self.get_loss_arb(w = w_hat_c4c5, ttm = data_dict["ttm_c4c5:0"], logm = data_dict["logm_c4c5:0"])
        w_hat_c6 = self.compute_w_hat(data_dict["ttm_c6:0"],data_dict["logm_c6:0"])
        c6 = self.get_loss_arb(w=w_hat_c6,ttm=data_dict["ttm_c6:0"],logm=data_dict["logm_c6:0"])
        return {
            "loss_fit_rmse":loss_fit["l_fit_iv_rmse"],
            "loss_fit_mape":loss_fit["l_fit_iv_mape"],
            "loss_c4":c4c5["l_c4"]+c6["l_c4"],
            "loss_c5":c4c5["l_c5"]+c6["l_c5"],
            "loss_c6":c6["l_c6"]
        }"""
    def get_test_loss(self,data_dict,pred_type="iv"):
        # 默认是iv的损失
        iv_hat_fit = self.compute_iv_hat(data_dict["ttm_fit:0"],data_dict["logm_fit:0"])
        iv_fit = data_dict["iv:0"]
        loss_fit = self.get_loss_fit_iv(iv_fit,iv_hat_fit)
        c6_num = len(data_dict["ttm_c6:0"])
        ttm_c4c5c6 = torch.cat((data_dict["ttm_c4c5:0"],data_dict["ttm_c6:0"]),dim=0)
        logm_c4c5c6 = torch.cat((data_dict["logm_c4c5:0"],data_dict["logm_c6:0"]),dim=0)
        arb_loss = self.get_loss_arb(ttm=ttm_c4c5c6,logm=logm_c4c5c6,c6_num=c6_num)
        return {
            "loss_fit_rmse":loss_fit["l_fit_iv_rmse"],
            "loss_fit_mape":loss_fit["l_fit_iv_mape"],
            "loss_c4":arb_loss["l_c4"],
            "loss_c5":arb_loss["l_c5"],
            "loss_c6":arb_loss["l_c6"]
        }
