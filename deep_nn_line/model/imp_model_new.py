import time
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def calc_loss(log_p, logdet, image_size, data_channel):
    n_pixel = image_size * data_channel
    loss = logdet + log_p
    return (
        ((-loss / n_pixel).mean()),
        (log_p / n_pixel).mean(),
        (logdet / n_pixel).mean(),
    )

@torch.jit.script
def logit_transform(x, constraint=torch.tensor(0.9)):
    # 只需要正向转换
    B, C, L = x.size()
    noise = torch.rand(B, C, L, device=x.device)
    x = (x * 255. + noise) / 256.
    x = x.sub_(1).mul_(constraint).add_(1).div_(2)
    
    logit_x = torch.log(x) - torch.log(1. - x)
    pre_logit_scale = torch.tensor(torch.log(constraint) - torch.log(1. - constraint), device=x.device)
    log_diag_J = F.softplus(logit_x) + F.softplus(-logit_x) - F.softplus(-pre_logit_scale)
    return logit_x, torch.sum(log_diag_J, dim=(1, 2))
    
class ImpModel(nn.Module):
    def __init__(self, neurons_vec=[40,40,40,40], activation="relu", data_channel=1,
                 noise_std=0.01, penalty=dict(fit=1, c4=10, c5=10, c6=10, atm=0.1), is_layernorm=True):
        super().__init__()
        self.data_channel = data_channel
        self.noise_std = noise_std
        self.penalty = penalty
        self.is_layernorm = is_layernorm
        
        # 激活函数选择
        activation_modules = {
            "relu": nn.ReLU,
            "softplus": nn.Softplus,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "selu": nn.SELU
        }
        self.afun = activation_modules[activation]()
        # 网络结构构建
        layers = []
        n_input = 2
        for n_neurons in neurons_vec:
            layer = nn.Linear(n_input, n_neurons)
            nn.init.kaiming_normal_(layer.weight)
            nn.init.normal_(layer.bias, std=1e-6)
            layers.append(layer)
            if is_layernorm:
                layers.append(nn.LayerNorm(n_neurons))
            layers.append(self.afun)
            n_input = n_neurons
        self.net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(n_input, 1)
        self.alpha = nn.Parameter(torch.tensor(1.0))
    def forward(self, ttm, logm):
        x = torch.stack((ttm, logm), dim=-1)
        x = self.net(x)
        return (1 + torch.tanh(self.output_layer(x))).flatten()
    
    """def compute_iv_hat(self,ttm,logm):
        return self.forward(ttm+1e-6,logm)"""
    
    """def compute_w_hat(self,ttm,logm):
        iv_hat = self.compute_iv_hat(ttm,logm)
        w_hat = iv_hat**2*(ttm+1e-6)
        return w_hat"""
    
    def compute_w_hat(self,ttm,logm):
        return self.forward(ttm,logm)

    def compute_iv_hat(self,ttm,logm):
        w_hat = self.compute_w_hat(ttm,logm)
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
    
    def get_loss_fit_old(self, w, w_hat, iv, iv_hat,iv_spread=None):
        # 减少计算次数，只计算一个标准下的损失
        # 目前先以iv为例
        # l_fit_w_rmse = torch.mean((1e-6 + (w - w_hat) ** 2) ** 0.5)
        l_fit_w_rmse = torch.sqrt(torch.mean(((w-w_hat)**2)))
        l_fit_w_mape = torch.mean(torch.abs((w_hat - w) / (w + 1e-6)))
        l_fit_w = (l_fit_w_rmse + l_fit_w_mape)
        
        l_fit_iv_mape = torch.mean(torch.abs((iv_hat - iv) / (iv + 1e-6)))

        if iv_spread is None:
            # l_fit_iv_rmse = torch.mean((1e-6 + (iv - iv_hat) ** 2) ** 0.5)
            l_fit_iv_rmse = torch.sqrt(torch.mean(((iv - iv_hat) ** 2)))
        else:
            l_fit_iv_rmse = torch.mean(1e-6 + torch.abs(iv - iv_hat) / (1 + iv_spread))
        
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
        """iv_hat_fit = self.compute_iv_hat(data_dict["ttm_fit:0"],data_dict["logm_fit:0"])
        w_hat_fit = iv_hat_fit**2*(data_dict["ttm_fit:0"]+1e-6)"""
        w_hat_fit = self.compute_w_hat(data_dict["ttm_fit:0"],data_dict["logm_fit:0"])
        iv_hat_fit = torch.sqrt(w_hat_fit / (data_dict["ttm_fit:0"]+1e-6))
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
        return self.penalty["fit"]*loss_fit+self.penalty["c4"]*arb_loss["l_c4"]+self.penalty["c5"]*arb_loss["l_c5"]+self.penalty["c6"]*arb_loss["l_c6"]

    # 生成数据网络
    def gen_iv_grid(self, ttm_s=None, logm_s=None, ttm_num=32, logm_num=32):
        device = next(self.parameters()).device
        ttm_s = torch.linspace(0.001, 1, ttm_num, device=device) if ttm_s is None else ttm_s
        logm_s = torch.linspace(-0.5, 0.5, logm_num, device=device) if logm_s is None else logm_s
        ttm_num,logm_num = len(ttm_s),len(logm_s)
        # 使用广播机制优化网格生成
        logm_grid, ttm_grid = torch.meshgrid(logm_s, ttm_s, indexing='xy')
        return logm_grid,ttm_grid,ttm_num,logm_num
    
    def get_surface_loss(self,data_dict,feed_model,logm_grid,ttm_grid,ttm_num,logm_num,gamma=0.001,data_pre='logit'):
        iv_data = self.compute_iv_hat(ttm_grid.ravel(), logm_grid.ravel())
        # iv_data = iv_data.view(ttm_num, 1, logm_num).clamp_(0, 2)
        iv_data = iv_data.view(ttm_num,1,logm_num)
        surface_data = iv_data.expand(-1, self.data_channel, -1)
        if data_pre == "logit":
            surface_data,_ = logit_transform(surface_data)
        elif data_pre == "normal":
            surface_data = surface_data/(2+1e-5) - 0.5
            surface_data += torch.rand_like(surface_data)*self.noise_std
        # 处理跨设备情况：将数据移到feed_model所在设备
        feed_device = next(feed_model.parameters()).device
        surface_data_feed = surface_data.to(feed_device)
        log_p,logdet,_ = feed_model(surface_data_feed)
        # 将结果移回主设备
        log_p = log_p.to(surface_data.device)
        logdet = logdet.to(surface_data.device)
        logdet.mean()
        loss_gen, log_p, log_det = calc_loss(log_p, logdet, torch.tensor(logm_num),torch.tensor(self.data_channel))
        fit_arb_loss = self.get_train_loss(data_dict)
        all_loss = gamma*loss_gen+fit_arb_loss
        return all_loss,loss_gen,fit_arb_loss
    
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

