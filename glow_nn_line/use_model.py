"""
使用训练好的 Glow 模型生成波动率曲线
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import platform

from gen_model import Glow
import data_utils


# 配置中文字体
def setup_chinese_font():
    """配置 matplotlib 中文字体"""
    system = platform.system()

    if system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'SimHei']
    elif system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']

    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_model(checkpoint_path, line_size=256, n_flow=4, n_block=2,
               filter_size=256, data_channel=1, device='cpu'):
    """
    加载训练好的 Glow 模型

    Args:
        checkpoint_path: 模型权重文件路径（.pt）
        line_size: 曲线长度
        n_flow: 每个block的flow数量
        n_block: block数量
        filter_size: 滤波器大小
        data_channel: 数据通道数
        device: 运行设备

    Returns:
        model: 加载好的模型（eval模式）
    """
    print(f"加载模型: {checkpoint_path}")

    # 创建模型结构
    model = Glow(
        in_channel=data_channel,
        n_flow=n_flow,
        n_block=n_block,
        length=line_size,
        filter_size=filter_size,
        affine=False,  # 根据训练参数调整
        conv_lu=True
    )

    # 加载权重
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # 设置为评估模式

    print("✅ 模型加载成功")
    return model


def calc_z_shapes(n_channel, length_size, n_block):
    """计算隐变量形状"""
    z_shapes = []
    for _ in range(n_block - 1):
        length_size //= 2
        z_shapes.append((n_channel, length_size))
    length_size //= 2
    z_shapes.append((n_channel * 2, length_size))
    return z_shapes


def generate_curves(model, n_samples=100, line_size=256, n_block=2,
                    data_channel=1, temperature=0.7, device='cpu'):
    """
    使用 Glow 模型生成波动率曲线

    Args:
        model: 训练好的 Glow 模型
        n_samples: 生成的曲线数量
        line_size: 曲线长度
        n_block: block数量
        data_channel: 数据通道数
        temperature: 采样温度（越大越随机，越小越确定）
        device: 设备

    Returns:
        curves: 生成的曲线 (n_samples, line_size)
    """
    print(f"生成 {n_samples} 条曲线...")

    with torch.no_grad():
        # 准备隐变量（从标准正态分布采样）
        z_sample = []
        z_shapes = calc_z_shapes(data_channel, line_size, n_block)

        for z_shape in z_shapes:
            z_new = torch.randn(n_samples, *z_shape) * temperature
            z_sample.append(z_new.to(device))

        # 生成曲线（反向传播）
        # MPS 设备需要在 CPU 上执行 reverse
        if device.type == 'mps':
            z_sample_cpu = [z.cpu() for z in z_sample]
            model.cpu()
            iv_surface = model.reverse(z_sample_cpu)
            model.to(device)
        else:
            iv_surface = model.reverse(z_sample)

        # 逆 logit 变换，还原到波动率空间
        iv_surface, _ = data_utils.logit_transform(iv_surface.cpu(), reverse=True)

        # 取通道均值
        iv_surface = torch.mean(iv_surface, dim=1, keepdim=True)
        curves = iv_surface.data.reshape(n_samples, line_size).numpy()

    print(f"✅ 生成完成")
    print(f"   曲线形状: {curves.shape}")
    print(f"   波动率范围: [{curves.min():.4f}, {curves.max():.4f}]")

    return curves


def save_curves(curves, output_path):
    """保存生成的曲线"""
    np.savetxt(output_path, curves, fmt='%.6f', delimiter=',')
    print(f"💾 曲线已保存到: {output_path}")


def visualize_curves(curves, n_display=10, z_min=-0.5, z_max=0.5,
                     save_path=None):
    """
    可视化生成的曲线

    Args:
        curves: 生成的曲线数组 (n_samples, line_size)
        n_display: 显示的曲线数量
        z_min, z_max: z值范围
        save_path: 保存图片路径（可选）
    """
    z_grid = np.linspace(z_min, z_max, curves.shape[1])

    plt.figure(figsize=(14, 6))

    # 子图1: 显示前 n_display 条曲线
    plt.subplot(1, 2, 1)
    for i in range(min(n_display, len(curves))):
        plt.plot(z_grid, curves[i], alpha=0.6, label=f'Curve {i+1}')
    plt.xlabel('z (log-moneyness / √T)', fontsize=12)
    plt.ylabel('Implied Volatility', fontsize=12)
    plt.title(f'前 {n_display} 条生成曲线', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=8)

    # 子图2: 所有曲线的统计分布
    plt.subplot(1, 2, 2)
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0)

    plt.plot(z_grid, mean_curve, 'b-', linewidth=2, label='均值')
    plt.fill_between(z_grid,
                     mean_curve - std_curve,
                     mean_curve + std_curve,
                     alpha=0.3, label='±1 标准差')
    plt.fill_between(z_grid,
                     mean_curve - 2*std_curve,
                     mean_curve + 2*std_curve,
                     alpha=0.2, label='±2 标准差')

    plt.xlabel('z (log-moneyness / √T)', fontsize=12)
    plt.ylabel('Implied Volatility', fontsize=12)
    plt.title(f'所有曲线的统计特征 (N={len(curves)})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 图片已保存到: {save_path}")

    plt.show()


def compare_with_real_data(generated_curves, real_data_path, z_min=-0.5, z_max=0.5):
    """
    对比生成曲线和真实数据

    Args:
        generated_curves: 生成的曲线
        real_data_path: 真实数据路径（可以是原始期权数据或SVI生成的曲线）
        z_min, z_max: z值范围
    """
    print(f"\n对比真实数据: {real_data_path}")

    # 读取真实数据
    try:
        real_curves = pd.read_csv(real_data_path, header=None).values
        print(f"   真实数据形状: {real_curves.shape}")
    except:
        print("   ⚠️  无法读取真实数据")
        return

    z_grid = np.linspace(z_min, z_max, generated_curves.shape[1])

    plt.figure(figsize=(12, 5))

    # 子图1: 真实数据
    plt.subplot(1, 2, 1)
    for i in range(min(5, len(real_curves))):
        plt.plot(z_grid, real_curves[i], alpha=0.5)
    plt.xlabel('z')
    plt.ylabel('IV')
    plt.title('真实数据（样本）')
    plt.grid(True, alpha=0.3)

    # 子图2: 生成数据
    plt.subplot(1, 2, 2)
    for i in range(min(5, len(generated_curves))):
        plt.plot(z_grid, generated_curves[i], alpha=0.5)
    plt.xlabel('z')
    plt.ylabel('IV')
    plt.title('生成数据（样本）')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 统计对比
    print(f"\n统计对比:")
    print(f"   真实数据 - 均值: {real_curves.mean():.4f}, 标准差: {real_curves.std():.4f}")
    print(f"   生成数据 - 均值: {generated_curves.mean():.4f}, 标准差: {generated_curves.std():.4f}")


def main():
    # 配置中文字体
    setup_chinese_font()

    parser = argparse.ArgumentParser(description="使用训练好的Glow模型生成波动率曲线")

    # 模型参数
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="模型权重路径（.pt文件）")
    parser.add_argument("--line_size", type=int, default=256,
                       help="曲线长度")
    parser.add_argument("--n_flow", type=int, default=4,
                       help="每个block的flow数量")
    parser.add_argument("--n_block", type=int, default=2,
                       help="block数量")
    parser.add_argument("--filter_size", type=int, default=256,
                       help="滤波器大小")

    # 生成参数
    parser.add_argument("--n_samples", type=int, default=100,
                       help="生成的曲线数量")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="采样温度（0.5-1.0，越小越确定）")
    parser.add_argument("--z_min", type=float, default=-0.5,
                       help="z值最小值")
    parser.add_argument("--z_max", type=float, default=0.5,
                       help="z值最大值")

    # 输出参数
    parser.add_argument("--output", type=str, default="./generated_curves.csv",
                       help="输出CSV文件路径")
    parser.add_argument("--plot", action="store_true",
                       help="是否可视化")
    parser.add_argument("--plot_save", type=str, default=None,
                       help="保存图片路径")
    parser.add_argument("--compare", type=str, default=None,
                       help="真实数据路径（用于对比）")

    # 设备参数
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda", "mps"],
                       help="运行设备")

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 加载模型
    model = load_model(
        args.checkpoint,
        line_size=args.line_size,
        n_flow=args.n_flow,
        n_block=args.n_block,
        filter_size=args.filter_size,
        device=device
    )

    # 生成曲线
    curves = generate_curves(
        model,
        n_samples=args.n_samples,
        line_size=args.line_size,
        n_block=args.n_block,
        temperature=args.temperature,
        device=device
    )

    # 保存结果
    save_curves(curves, args.output)

    # 可视化
    if args.plot:
        visualize_curves(
            curves,
            n_display=10,
            z_min=args.z_min,
            z_max=args.z_max,
            save_path=args.plot_save
        )

    # 对比真实数据
    if args.compare:
        compare_with_real_data(curves, args.compare, args.z_min, args.z_max)

    print("\n✅ 完成！")


if __name__ == "__main__":
    main()