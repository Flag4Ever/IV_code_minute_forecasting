from functools import partial

import numpy as np
from numpy import ndarray, array, arange, zeros, ones, argmin, minimum, maximum, clip
from numpy.linalg import norm
from numpy.random import normal
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# 面向对象的编程，类在初始化的时候应该实现什么内容？
# 这里先看一下是否采用vcr/scr/ssr的默认设置
class WingModel:
    def __init__(self) -> None:
        # 采用一般设置！vcr=0,scr=0,ssr=100
        # 上述参数应该是由于atm和ref不同所带来的参数
        pass

    def skew(self, moneyness: ndarray, vc: float, sc: float, pc: float, cc: float, dc: float, uc: float, dsm: float,
             usm: float) -> ndarray:
        # 确定vc/sc/pc/cc/dc/uc/dsm/usm后的计算过程
        # 这几个参数也是我们要拟合的参数,因为atm参数在创建数据集的时候就已经计算出来了
        if vc < 1e-6:
            vc = 1e-6
        elif vc > 10:
            vc = 10
        assert -1 < dc < 0
        assert dsm > 0
        assert 1 > uc > 0
        assert usm > 0
        assert 1e-6 <= vc <= 10  # 数值优化过程稳定
        assert -1e6 < sc < 1e6
        assert dc * (1 + dsm) <= dc <= 0 <= uc <= uc * (1 + usm)

        # volatility at this converted strike, vol(x) is then calculated as follows:
        vol_list = []
        for x in moneyness:
            # volatility at this converted strike, vol(x) is then calculated as follows:
            if x < dc * (1 + dsm):
                # 第一个区域
                # 当x特别小的时候，其对应的结果为常数
                vol = vc + dc * (2 + dsm) * (sc / 2) + (1 + dsm) * pc * pow(dc, 2)
            elif dc * (1 + dsm) < x <= dc:
                # 第二个区域
                vol = vc - (1 + 1 / dsm) * pc * pow(dc, 2) - sc * dc / (2 * dsm) + (1 + 1 / dsm) * (
                        2 * pc * dc + sc) * x - (pc / dsm + sc / (2 * dc * dsm)) * pow(x, 2)
            elif dc < x <= 0:
                # 第三个区域
                vol = vc + sc * x + pc * pow(x, 2)
            elif 0 < x <= uc:
                # 第四个区域
                vol = vc + sc * x + cc * pow(x, 2)
            elif uc < x <= uc * (1 + usm):
                # 第五个区域
                vol = vc - (1 + 1 / usm) * cc * pow(uc, 2) - sc * uc / (2 * usm) + (1 + 1 / usm) * (
                        2 * cc * uc + sc) * x - (cc / usm + sc / (2 * uc * usm)) * pow(x, 2)
            elif uc * (1 + usm) < x:
                # 第六个区域，其值也是常数
                vol = vc + uc * (2 + usm) * (sc / 2) + (1 + usm) * cc * pow(uc, 2)
            else:
                raise ValueError("x value error!")
            vol_list.append(vol)
            # 最后返回的vol_list收集了不同x对应的隐含波动率值
        return array(vol_list)
    
    def loss_skew(self, params:list, x: ndarray, iv: ndarray, vega: ndarray, vc: float, dc: float,
                  uc: float, dsm: float, usm: float):
        # 实际上确定拟合范围的参数为：dc/dsm/uc/usm,而vc是x=0时的纵坐标,可以用函数进行拟合
        # 因此，需要拟合的参数只有sc, pc, cc三个浮点数
        # 暂时先按照这样的形式使用
        sc, pc, cc = params
        vega = vega / vega.max()
        # vega是在损失上面加上的权重
        value = self.skew(x, vc, sc, pc, cc, dc, uc, dsm, usm)
        # 损失是求平方后再开根号
        return norm((value - iv) * vega, ord=2, keepdims=False)
    
    def calibrate_skew(self, x: ndarray, iv: ndarray, vega: ndarray, dc: float = -0.2, uc: float = 0.2, dsm: float = 0.5,
                       usm: float = 0.5, is_bound_limit: bool = False,
                       epsilon: float = 1e-16, inter: str = "cubic"):
        # 参数校准过程
        # vc是x=0时的取值，因此采用函数进行拟合
        vc = interp1d(x, iv, kind=inter, fill_value="extrapolate")([0])[0]
        # init guess for sc, pc, cc
        if is_bound_limit:
            bounds = [(-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3)]
        else:
            bounds = [(None, None), (None, None), (None, None)]
        initial_guess = normal(size=3)
        args = (x, iv, vega, vc, dc, uc, dsm, usm)
        # 这里应该只是对sc,pc,cc三个参数进行了校准
        residual = minimize(self.loss_skew, initial_guess, args=args, bounds=bounds, tol=epsilon, method="SLSQP")
        assert residual.success
        # 包含最佳结果和最佳参数点
        return residual.x, residual.fun
    
    def loss_test(self, params:list, x: ndarray, iv: ndarray, vc: float, dc: float,
                  uc: float, dsm: float, usm: float):
        sc, pc, cc = params
        value = self.skew(x, vc, sc, pc, cc, dc, uc, dsm, usm)
        mse = np.mean((value-iv)**2)
        return {
            "mse":mse,
            "rmse":np.sqrt(mse),
            "mape":np.mean(np.abs((value - iv) / (iv + 1e-6)))
        }

# 考虑模型还需要包含一定的无套利条件

class ArbitrageFreeWingModel(WingModel):
    # 参数校准过程
    def __init__(self) -> None:
        super().__init__()

    def calibrate(self, x: ndarray, iv: ndarray, vega: ndarray, dc: float = -0.2, uc: float = 0.2, dsm: float = 0.5,
                  usm: float = 0.5, is_bound_limit: bool = False, epsilon: float = 1e-16, inter: str = "cubic",
                  level: float = 1, method: str = "SLSQP", epochs = None, show_error: bool = False,
                  use_constraints: bool = False):
        """
        # 这里对三个参数进行了校准
        其他的参数取默认值
        """
        vega = clip(vega, 1e-6, 1e6)
        iv = clip(iv, 1e-6, 10)

        # init guess for sc, pc, cc
        if is_bound_limit:
            bounds = [(-1e3, 1e3), (-1e3, 1e3), (-1e3, 1e3)]
        else:
            bounds = [(None, None), (None, None), (None, None)]

        vc = interp1d(x, iv, kind=inter, fill_value="extrapolate")([0])[0]
        constraints = dict(type='ineq', fun=partial(self.constraints, args=(x, vc, dc, uc, dsm, usm), level=level))
        args = (x, iv, vega, vc, dc, uc, dsm, usm)
        # loss_skew:
        # loss_skew(cls, params:list, x: ndarray,
        #           iv: ndarray, vega: ndarray, vc: float, dc: float,
        #           uc: float, dsm: float, usm: float)
        # 
        if epochs is None:
            if use_constraints:
                # 假如使用约束的话,将约束加入minimize过程
                residual = minimize(self.loss_skew, normal(size=3), args=args, bounds=bounds, constraints=constraints,
                                    tol=epsilon, method=method)
            else:
                residual = minimize(self.loss_skew, normal(size=3), args=args, bounds=bounds, tol=epsilon, method=method)

            if residual.success:
                sc, pc, cc = residual.x
                arbitrage_free = self.check_butterfly_arbitrage(sc, pc, cc, dc, dsm, uc, usm, x, vc)
                return residual.x, residual.fun, arbitrage_free
            else:
                epochs = 10
                if show_error:
                    print("calibrate wing-model wrong, use epochs = 10 to find params! params: {}".format(residual.x))

        if epochs is not None:
            params = zeros([epochs, 3])
            loss = ones([epochs, 1])
            for i in range(epochs):
                if use_constraints:
                    residual = minimize(self.loss_skew, normal(size=3), args=args, bounds=bounds,
                                        constraints=constraints,
                                        tol=epsilon, method="SLSQP")
                else:
                    residual = minimize(self.loss_skew, normal(size=3), args=args, bounds=bounds, tol=epsilon,
                                        method="SLSQP")
                if not residual.success and show_error:
                    print("calibrate wing-model wrong, wrong @ {} /{}! params: {}".format(i,epochs,residual.x))
                params[i] = residual.x
                loss[i] = residual.fun
            min_idx = argmin(loss)
            sc, pc, cc = params[min_idx]
            loss = loss[min_idx][0]
            arbitrage_free = self.check_butterfly_arbitrage(sc, pc, cc, dc, dsm, uc, usm, x, vc)
            return (sc, pc, cc), vc, loss, arbitrage_free
    
    def constraints(self, x, args,
                    level: float = 1) -> float:
        """蝶式价差无套利约束
        取100个点进行无套利的约束
        :param x: guess values, sc, pc, cc
        :param args:[ndarray, float, float, float, float, float]
        :param level:
        :return:
        """
        sc, pc, cc = x
        moneyness, vc, dc, uc, dsm, usm = args

        if level == 0:
            pass
        elif level == 1:
            moneyness = arange(-1.5, 0.5, 0.01)
        else:
            moneyness = arange(-1.5, 0.5, 0.001)
        # 计算蝶式价差的大小
        return self.check_butterfly_arbitrage(sc, pc, cc, dc, dsm, uc, usm, moneyness, vc)

    def left_parabolic(self,sc: float, pc: float, x: float, vc: float) -> float:
        """
        左侧的抛物线
        对应区域3
        """
        return pc - 0.25 * (sc + 2 * pc * x) ** 2 * (0.25 + 1 / (vc + sc * x + pc * x * x)) + (
                1 - 0.5 * x * (sc + 2 * pc * x) / (vc + sc * x + pc * x * x)) ** 2
    
    def right_parabolic(self,sc: float, cc: float, x: float, vc: float) -> float:
        """
        右侧的抛物线
        对应区域4
        """
        return cc - 0.25 * (sc + 2 * cc * x) ** 2 * (0.25 + 1 / (vc + sc * x + cc * x * x)) + (
                1 - 0.5 * x * (sc + 2 * cc * x) / (vc + sc * x + cc * x * x)) ** 2
    
    def left_smoothing_range(self, sc: float, pc: float, dc: float, dsm: float, x: float, vc: float) -> float:
        # 对应区域2
        a = - pc / dsm - 0.5 * sc / (dc * dsm)
        b1 = -0.25 * ((1 + 1 / dsm) * (2 * dc * pc + sc) - 2 * (pc / dsm + 0.5 * sc / (dc * dsm)) * x) ** 2
        b2 = -dc ** 2 * (1 + 1 / dsm) * pc - 0.5 * dc * sc / dsm + vc + (1 + 1 / dsm) * (2 * dc * pc + sc) * x - (
                pc / dsm + 0.5 * sc / (dc * dsm)) * x ** 2
        b2 = (0.25 + 1 / b2)
        b = b1 * b2

        c1 = x * ((1 + 1 / dsm) * (2 * dc * pc + sc) - 2 * (pc / dsm + 0.5 * sc / (dc * dsm)) * x)
        c2 = 2 * (-dc ** 2 * (1 + 1 / dsm) * pc - 0.5 * dc * sc / dsm + vc + (1 + 1 / dsm) * (2 * dc * pc + sc) * x - (
                pc / dsm + 0.5 * sc / (dc * dsm)) * x ** 2)
        c = (1 - c1 / c2) ** 2
        return a + b + c
    
    def right_smoothing_range(self, sc: float, cc: float, uc: float, usm: float, x: float, vc: float) -> float:
        # 对应区域5
        a = - cc / usm - 0.5 * sc / (uc * usm)
        b1 = -0.25 * ((1 + 1 / usm) * (2 * uc * cc + sc) - 2 * (cc / usm + 0.5 * sc / (uc * usm)) * x) ** 2
        b2 = -uc ** 2 * (1 + 1 / usm) * cc - 0.5 * uc * sc / usm + vc + (1 + 1 / usm) * (2 * uc * cc + sc) * x - (
                cc / usm + 0.5 * sc / (uc * usm)) * x ** 2
        b2 = (0.25 + 1 / b2)
        b = b1 * b2

        c1 = x * ((1 + 1 / usm) * (2 * uc * cc + sc) - 2 * (cc / usm + 0.5 * sc / (uc * usm)) * x)
        c2 = 2 * (-uc ** 2 * (1 + 1 / usm) * cc - 0.5 * uc * sc / usm + vc + (1 + 1 / usm) * (2 * uc * cc + sc) * x - (
                cc / usm + 0.5 * sc / (uc * usm)) * x ** 2)
        c = (1 - c1 / c2) ** 2
        return a + b + c
    
    def left_constant_level(self) -> float:
        # 对应区域1,即常数部分
        return 1


    def right_constant_level(self) -> float:
        # 对应区域6,常数部分
        return 1
    
    def _check_butterfly_arbitrage(self, sc: float, pc: float, cc: float, dc: float, dsm: float, uc: float, usm: float,
                                   x: float, vc: float) -> float:
        if x < dc * (1 + dsm):
            return self.left_constant_level()
        elif dc * (1 + dsm) < x <= dc:
            return self.left_smoothing_range(sc, pc, dc, dsm, x, vc)
        elif dc < x <= 0:
            return self.left_parabolic(sc, pc, x, vc)
        elif 0 < x <= uc:
            return self.right_parabolic(sc, cc, x, vc)
        elif uc < x <= uc * (1 + usm):
            return self.right_smoothing_range(sc, cc, uc, usm, x, vc)
        elif uc * (1 + usm) < x:
            return self.right_constant_level()
        else:
            raise ValueError("x value error!")

        """
        检查是否存在蝶式价差套利机会，确保拟合time-slice iv-curve 是无套利（无蝶式价差静态套利）曲线
        这里只计算了在左右抛物线时的蝶式价差套利
        if dc < x <= 0:
            return self.left_parabolic(sc, pc, x, vc)
        elif 0 < x <= uc:
            return self.right_parabolic(sc, cc, x, vc)
        else:
            return 0"""
    
    def check_butterfly_arbitrage(self, sc: float, pc: float, cc: float, dc: float, dsm: float, uc: float, usm: float,
                                  moneyness: ndarray, vc: float) -> float:
        con_arr = []
        for x in moneyness:
            con_arr.append(self._check_butterfly_arbitrage(sc, pc, cc, dc, dsm, uc, usm, x, vc))
        con_arr = array(con_arr)
        if (con_arr >= 0).all():
            return minimum(con_arr.mean(), 1e-7)
        else:
            return maximum((con_arr[con_arr < 0]).mean(), -1e-7)
    
    def get_but_loss(self, sc: float, pc: float, cc: float, dc: float, dsm: float, uc: float, usm: float,
                                  moneyness: ndarray, vc: float):
        con_arr = []
        for x in moneyness:
            con_arr.append(self._check_butterfly_arbitrage(sc, pc, cc, dc, dsm, uc, usm, x, vc))
        con_arr = array(con_arr)
        # print(len(con_arr))
        return {
            "l_butterfly": np.maximum(-con_arr, 0).sum(),
            "butterfly": con_arr.sum(),
        }
