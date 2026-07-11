# 生成一个"分布外 + 谱密集杂乱"的测试信号 (与训练集刻意不同)
#
# 训练集特征: 合成信号是 3~10 个有最小间距、相对干净的 AM-FM 峰; 风电是单条序列。
# 本脚本刻意塞进训练分布里没有的成分, 让频谱连续、杂乱、难分:
#   1) 谐波梳   — 基频 + 多次谐波 (成排的尖峰, 间距远小于训练 min_spacing)
#   2) 扫频啁啾 — 频率随时间线性/反向扫动 (训练只有固定中心频率的 FM)
#   3) 拍频对   — 两个靠得极近的音 (间距 < 模型分辨率, 故意制造混叠难点)
#   4) 瞬态脉冲包 — 高斯窗高频包 (时频局部化, 谱上是展宽的鼓包)
#   5) 有色(粉红)噪声铺底 — 1/sqrt(f) 谱, 让整条频谱连续而非孤立峰
#   6) 白噪声地板
# 全部用【归一化频率】(cycles/sample) 设计, 与模型一致; 长度默认 1024 对齐训练窗口。
#
# 用法:
#   cd Code && python make_complex_signal.py                       # 只生成 + 预览谱
#   cd Code && python make_complex_signal.py --checkpoint ../exp/vmd_wind/best_model.pth.tar
#                                                                  # 顺带跑分解

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# 复杂测试信号生成 (归一化频率, cycles/sample)
# ============================================================================

def make_complex_signal(N=1024, seed=20260628):
    """返回一个长度 N、零均值单位方差的复杂多分量信号 (np.float32)。

    频率全部以归一化频率 f (cycles/sample) 给出, 有效带 (0, 0.5)。
    """
    rng = np.random.RandomState(seed)
    n = np.arange(N, dtype=np.float64)
    x = np.zeros(N, dtype=np.float64)

    def tone(f, amp=1.0, phase=0.0):
        return amp * np.sin(2 * np.pi * f * n + phase)

    def am(depth, fmod):
        # 慢幅度调制 (fmod 用极低归一化频率, 几个周期/全窗)
        return 1.0 + depth * np.sin(2 * np.pi * fmod * n + rng.uniform(0, 2 * np.pi))

    def chirp(f0, f1, amp=1.0):
        # 瞬时频率从 f0 线性扫到 f1; 相位 = 2π·累积瞬时频率
        inst_f = np.linspace(f0, f1, N)
        phase = 2 * np.pi * np.cumsum(inst_f)
        return amp * np.sin(phase)

    def burst(fc, center, width, amp=1.0):
        # 高斯窗高频包: 时域局部、频域展宽
        env = np.exp(-0.5 * ((n - center) / width) ** 2)
        return amp * env * np.sin(2 * np.pi * fc * n + rng.uniform(0, 2 * np.pi))

    # 1) 谐波梳: 基频 0.03 + 2~6 次谐波, 幅度递减 (成排尖峰, 间距 0.03 << 训练 min_spacing)
    f0 = 0.03
    for k in range(1, 7):
        x += tone(k * f0, amp=1.0 / k) * am(0.3, rng.uniform(0.002, 0.006))

    # 2) 若干带 AM 的平稳音 (分布在中高频, 与谐波梳错开)
    for f in [0.105, 0.165, 0.225, 0.305, 0.385, 0.445]:
        x += tone(f, amp=rng.uniform(0.5, 1.2), phase=rng.uniform(0, 2 * np.pi)) \
             * am(rng.uniform(0.2, 0.5), rng.uniform(0.002, 0.008))

    # 3) 拍频对: 两个靠得极近的音 (间距 0.006, 接近/低于模型分辨率 1/256≈0.0039)
    x += tone(0.255, amp=0.8) + tone(0.261, amp=0.8)

    # 4) 三条扫频啁啾 (上扫 / 上扫 / 下扫), 横跨频带
    x += chirp(0.04, 0.16, amp=0.7)
    x += chirp(0.30, 0.47, amp=0.6)
    x += chirp(0.22, 0.10, amp=0.6)

    # 5) 瞬态脉冲包 (高频, 时域局部) — 制造时频局部化的展宽鼓包
    x += burst(0.40, center=0.20 * N, width=0.03 * N, amp=1.5)
    x += burst(0.46, center=0.70 * N, width=0.025 * N, amp=1.3)

    # 6) 有色(粉红)噪声铺底: 白噪声经 1/sqrt(f) 整形 → 谱连续, 低频偏强
    white = rng.randn(N)
    spec = np.fft.rfft(white)
    f_axis = np.fft.rfftfreq(N)
    shape = np.ones_like(f_axis)
    shape[1:] = 1.0 / np.sqrt(f_axis[1:])     # 避免 DC 除零
    pink = np.fft.irfft(spec * shape, n=N)
    pink = pink / (pink.std() + 1e-8)
    x += 0.35 * pink

    # 7) 白噪声地板
    x += 0.05 * rng.randn(N)

    # 零均值单位方差 (与风电切片 / test_wind 预处理一致)
    x = x - x.mean()
    x = x / (x.std() + 1e-8)
    return x.astype(np.float32)


# ============================================================================
# 预览图: 时域 + 幅度谱 (归一化频率横轴)
# ============================================================================

def preview(signal, out_path):
    N = len(signal)
    win = np.hanning(N)
    f_axis = np.fft.rfftfreq(N)               # 归一化 [0, 0.5]
    sp = np.abs(np.fft.rfft(signal * win))

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 3.2))
    a0.plot(np.arange(N), signal, color='#1f77b4', linewidth=0.6)
    a0.set_title('Complex test signal — time'); a0.set_xlabel('sample')
    a0.set_ylabel('amplitude'); a0.grid(True, alpha=0.3); a0.set_xlim(0, N - 1)

    a1.plot(f_axis, sp, color='#d62728', linewidth=0.7)
    a1.set_title('Spectrum (normalized freq, cycles/sample)')
    a1.set_xlabel('normalized frequency'); a1.set_ylabel('magnitude')
    a1.grid(True, alpha=0.3); a1.set_xlim(0, 0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'预览图已保存: {out_path}')


# ============================================================================
# 可选: 加载 checkpoint 跑分解 (复用 test_wind 的 decompose + 绘图)
# ============================================================================

def run_decompose(signal, args):
    import torch
    from models import NeuralSVMD
    from test_wind import decompose, plot_all

    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    model = NeuralSVMD(n_fft=args.n_fft, hop_length=args.hop_length, hidden_dim=64,
                       sample_rate=args.sample_rate)
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print(f'模型: {args.checkpoint} (epoch {ckpt.get("epoch", "?")})')

    modes, cfs, residual = decompose(
        model, signal, device, args.sample_rate,
        max_steps=args.max_steps, epsilon=args.epsilon)
    print(f'提取模态数: {len(modes)}  CFs(Hz, ×sample_rate): {[f"{c:.4f}" for c in cfs]}')

    recon = np.sum(modes, axis=0) + residual
    print(f'重构 MSE: {np.mean((signal - recon) ** 2):.6e}')

    out = args.out.replace('.png', '_decomp.png')
    plot_all(signal, modes, cfs, residual, args.sample_rate, out, 'complex test signal')


def main():
    ap = argparse.ArgumentParser(description='生成分布外的复杂测试信号 (+可选分解)')
    ap.add_argument('--N', type=int, default=1024, help='信号长度 (对齐训练窗口)')
    ap.add_argument('--seed', type=int, default=20260628, help='随机种子 (换种子=换一条信号)')
    ap.add_argument('--out', default='../exp/wind/complex_test.png', help='预览图输出路径')
    ap.add_argument('--save_npy', default='../exp/wind/complex_test.npy', help='信号 .npy 保存路径')
    # 可选分解 (给了 --checkpoint 才跑)
    ap.add_argument('--checkpoint', default='', help='给定则加载并分解该信号')
    ap.add_argument('--n_fft', type=int, default=256)
    ap.add_argument('--hop_length', type=int, default=64)
    ap.add_argument('--sample_rate', type=float, default=0.0166667)
    ap.add_argument('--max_steps', type=int, default=20)
    ap.add_argument('--epsilon', type=float, default=0.005)
    ap.add_argument('--use_cuda', type=int, default=0)
    args = ap.parse_args()

    signal = make_complex_signal(args.N, args.seed)
    print(f'信号: 长度 {len(signal)}  均值 {signal.mean():.2e}  std {signal.std():.4f}  '
          f'范围 [{signal.min():.2f}, {signal.max():.2f}]')

    os.makedirs(os.path.dirname(os.path.abspath(args.save_npy)), exist_ok=True)
    np.save(args.save_npy, signal)
    print(f'信号已保存: {args.save_npy}')

    preview(signal, args.out)

    if args.checkpoint:
        run_decompose(signal, args)


if __name__ == '__main__':
    main()
