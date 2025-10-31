import numpy as np
from scipy.optimize import minimize

class sabr_vol:
    
    # def __init__(self,data,beta,shift,strike):
    def __init__(self,k,f,t,iv,beta,shift):
        # 取出行权价格
        # k:narray
        self.k=k
        # 取出远期价格
        # 一个模型的远期价格应该是确定的
        self.f=f
        # 取出到期期限
        # 到期期限也是定的
        self.t=t
        # 给定beta的值
        self.beta=beta
        self.v_sln=iv*100
        # shift也是为了数值稳定
        self.shift=shift
        self.alpha=None
        self.rho=None
        self.volvar=None
    
    def sabr_vol(self,k,alpha,rho,volvar):
        # 需要事先给定alpha的值
        # 用于定义隐含波动率的公式计算
        f=self.f
        # strike为负或forward price为负
        if k <= 0 or f <= 0:
            return 0        
        # f等于k的情况，用精度数代替0
        eps = 1e-7
        logfk = np.log(f/k)
        fk = (f*k)**(1-self.beta)
        a = (1-self.beta)**2*alpha**2/(24*fk)
        b = 0.25*rho*self.beta*volvar*alpha/fk**0.5
        c = (2-3*rho**2)*volvar**2/24
        d = fk**0.5
        v = (1-self.beta)**2*logfk**2/24
        w = (1-self.beta)**4*logfk**4/1920
        z = volvar*fk**0.5*logfk/alpha
        def x(rho, z):
            """
            返回函数x基于 sabr lognormal vol expansion
            """
            a = (1-2*rho*z+z**2)**0.5+z-rho
            b = 1-rho
            return np.log(a/b)
        if abs(z) > eps:
              vz = alpha * z * (1 + (a+b+c) * self.t) / (d * (1+v+w)*x(rho, z))
              return vz
        else:
            v0 = alpha*(1+(a+b+c)*self.t)/(d*(1+v+w))
            return v0
     
    def calibration(self,tol=1e-5,epochs=100):
        
        """
        校准sabr模型的参数alpha，rho，volvar
    
        基于bs model算出的volatility smile （strike和volatility两个维度）
        返回一个sabr模型的参数元组
        """
        
        def vol_square_error(x):
            
            vols = [self.sabr_vol(k_+self.shift,  x[0],
                                  x[1], x[2])*100 for k_ in self.k]
            return sum((vols-self.v_sln)**2)
        

        params = np.zeros([epochs, 3])
        loss = np.ones([epochs, 1])
        for i in range(epochs):
            x0 = np.array([0.01, 0.00, 0.1])
            bounds = [(0.0001, None), (-0.9999, 0.9999), (0.0001, None)]
            res = minimize(vol_square_error, x0, method='L-BFGS-B', bounds=bounds,tol=tol)
            if not res.success:
                print("calibrate sabr-model wrong, wrong @ {} /{}! params: {}".format(i,epochs,res.x))
            params[i] = res.x
            loss[i] = res.fun
        min_idx = np.argmin(loss)
        self.alpha, self.rho, self.volvar = params[min_idx]
        return [self.alpha, self.rho, self.volvar]
    
    def get_sabr_vol(self,strike):
        # strike:narray
        # 利用sabr的参数计算strike对应的隐含波动率
        if self.alpha is None:
            print("error:use model before calibration!")
        else:
            vols = [self.sabr_vol(k_+self.shift,self.alpha,
                                  self.rho, self.volvar) for k_ in strike]
        return vols
    
    def compute_iv_hat_derivative(self, logfk, dk=1e-6):
        # 只能计算蝶式期权的套利
        # 并且只能在特定的日期计算
        f = self.f
        k = f/np.exp(logfk)
        vols_plus = np.array([self.sabr_vol(k_ + dk +self.shift, self.alpha,self.rho, self.volvar) for k_ in k])
        vols_minus = np.array([self.sabr_vol(k_ - dk +self.shift, self.alpha,self.rho, self.volvar) for k_ in k])
        divdk = (vols_plus - vols_minus) / (2 * dk)
        return divdk
    
    def compute_iv_hat_second_derivative(self,logfk,dk=1e-6):
        f = self.f
        k = f/np.exp(logfk)
        vols_plus = np.array([self.sabr_vol(k_ + dk +self.shift, self.alpha,self.rho, self.volvar) for k_ in k])
        vols_center = np.array([self.sabr_vol(k_ +self.shift, self.alpha,self.rho, self.volvar) for k_ in k])
        vols_minus = np.array([self.sabr_vol(k_ - dk +self.shift, self.alpha,self.rho, self.volvar) for k_ in k])
        d2ivdk2 = (vols_plus - 2 * vols_center + vols_minus) / (dk**2)
        return d2ivdk2
    
    def get_arb_loss(self, logfk):
        """计算SABR模型的无套利条件损失"""
        f = self.f
        k = f/np.exp(logfk)
        # 计算一阶导数
        # print(len(k))
        divdk = self.compute_iv_hat_derivative(logfk)
        
        # 计算二阶导数
        d2ivdk2 = self.compute_iv_hat_second_derivative(logfk)
        
        # 计算蝶式套利条件
        iv_hat = np.array([self.sabr_vol(k_ +self.shift, self.alpha,self.rho, self.volvar) for k_ in k])
        # print(len(iv_hat))
        d1 = -k / (iv_hat * np.sqrt(self.t)) + (1/2) * iv_hat * np.sqrt(self.t)
        d2 = -k / (iv_hat * np.sqrt(self.t)) - (1/2) * iv_hat * np.sqrt(self.t)
        butterfly = (1 + d1 * divdk * np.sqrt(self.t)) * (1 + d2 * divdk * np.sqrt(self.t)) + iv_hat * d2ivdk2 * self.t
        return {
            "l_butterfly": np.maximum(-butterfly, 0).sum(),
            "butterfly": butterfly.sum(),
        }

    def loss_mse(self,sarb_iv,iv):
        # 如何计算所有的数值?
        return np.mean((sarb_iv-iv)**2)
    
    def loss_mape(self,sarb_iv,iv):
        return np.mean(np.abs((sarb_iv - iv) / (iv + 1e-6)))
    
    # 计算价格方面的误差函数
    def loss_price_mse(self,price_hat,price_target):
        return np.mean((price_hat-price_target)**2)
    
    def loss_price_mape(self,price_hat,price_target):
        return np.mean(np.abs((price_hat-price_target)/(price_target+1e-6)))
    
    def loss_price_spread(self,price_hat,price_target,price_ask,price_bid):
        s_ask_bid = price_ask - price_bid
        assert (s_ask_bid>0).all(),"有ask小于bid!"
        loss_spread = np.mean((2*np.abs(price_hat-price_target))/(s_ask_bid))
        return loss_spread
