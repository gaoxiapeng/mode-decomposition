# Neural SVMD — 推理与可视化 Demo
# 递归分解信号为多个模态分量，自适应停止

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import NeuralSVMD
from losses import compute_center_freq, compute_residual_energy


SAMPLE_RATE = 8000
N_FFT = 512
CMAP = 'viridis'


# ============================================================================
# 测试信号生成
# ============================================================================

def make_test_signal(sr=8000, duration=2.0):
    """生成一个多分量测试信号 (8 个模态)."""
    T = int(sr * duration)
    t = np.linspace(0, duration, T)
    np.random.seed(42)

    modes = []
    # 8 modes with varied frequencies (100–3800 Hz), modulations, and amplitudes
    configs = [
        (120,  0.3, 1.0,  'AM'),    # low freq AM
        (350,  0.0, 0.8,  'FM'),    # FM
        (600,  0.2, 0.7,  'AM'),    # mid AM
        (950,  0.0, 0.6,  'CW'),    # pure sine
        (1400, 0.4, 0.5,  'AM'),    # AM
        (2000, 0.0, 0.4,  'FM'),    # FM
        (2700, 0.1, 0.3,  'AM'),    # AM
        (3500, 0.0, 0.2,  'CW'),    # high pure sine
    ]

    for fc, mod_depth, amp, mod_type in configs:
        if mod_type == 'AM':
            mode = amp * np.sin(2 * np.pi * fc * t) * (1 + mod_depth * np.sin(2 * np.pi * np.random.uniform(3, 10) * t))
        elif mod_type == 'FM':
            mode = amp * np.sin(2 * np.pi * fc * t + mod_depth * 5 * np.cos(2 * np.pi * np.random.uniform(3, 12) * t))
        else:  # CW
            mode = amp * np.sin(2 * np.pi * fc * t)
        modes.append(mode.astype(np.float32))

    mixture = np.sum(modes, axis=0) + 0.02 * np.random.randn(T).astype(np.float32)
    return mixture.astype(np.float32), modes, sr


# ============================================================================
# 模型加载
# ============================================================================

def load_model(model_path, use_cuda=True):
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    model = NeuralSVMD(
        n_fft=512, hop_length=128, hidden_dim=64, sample_rate=SAMPLE_RATE
    )
    ckpt = torch.load(model_path, map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path} (epoch {ckpt.get('epoch', '?')})")
    return model, device


# ============================================================================
# 递归分解
# ============================================================================

@torch.no_grad()
def decompose(model, signal, device, max_steps=10, epsilon=0.001,
              plateau_ratio=0.02, dup_freq_hz=50.0):
    """
    递归分解信号为模态分量.

    残差直接喂下一步 (residual = input - mode), 全局重构精确成立.

    三重停止判据 (任一满足即停, 防止真实模态提完后产出重复/混叠伪模态):
      1. 能量停止 : 残差/原能量 < epsilon
      2. 平台停止 : 残差能量降幅 < plateau_ratio
      3. 重复停止 : 新模态中心频率与任一已提取模态接近 < dup_freq_hz

    Args:
        model: NeuralSVMD
        signal: [T] numpy array
        device: torch device
        max_steps: 最大递归步数
        epsilon: 残差能量阈值 (相对原始信号)
        plateau_ratio: 残差能量相对上一步的最小降幅
        dup_freq_hz: 判定为重复模态的中心频率间距 (Hz)

    Returns:
        modes: list of [T] numpy arrays
        center_freqs: list of floats (Hz)
        final_residual: [T] numpy array
    """
    signal_tensor = torch.from_numpy(signal).float().unsqueeze(0).to(device)  # [1, T]
    original_energy = torch.mean(signal_tensor ** 2).item()

    modes = []
    center_freqs = []
    current_input = signal_tensor
    prev_res_energy = original_energy

    for step in range(max_steps):
        output = model(current_input, step)  # dict

        # 调试：检查输出是否包含nan
        if torch.isnan(output["mode"]).any():
            print(f"  Step {step+1}: WARNING - Output contains NaN!")
            print(f"    Input shape: {current_input.shape}")
            print(f"    Input stats: mean={current_input.mean():.6f}, std={current_input.std():.6f}, min={current_input.min():.6f}, max={current_input.max():.6f}")
            print(f"    Mode stats: mean={output['mode'].mean():.6f}, std={output['mode'].std():.6f}")
            print(f"  ERROR: Model output contains NaN. Cannot continue decomposition.")
            return modes, center_freqs, np.zeros_like(signal)

        u_L = output["mode"][0, :].cpu().numpy()
        f_r = output["residual"]  # [1, T]

        # 谱质心
        cf = compute_center_freq(output["mode"], sample_rate=SAMPLE_RATE).item()

        # 残差能量
        residual_energy = compute_residual_energy(f_r).item()
        rel_energy = residual_energy / (original_energy + 1e-8)

        # 重复停止: 新模态与任一历史模态中心频率过近 → 混叠, 丢弃并停
        if center_freqs and min(abs(cf - c) for c in center_freqs) < dup_freq_hz:
            print(f"  -> Stopping: center_freq {cf:.1f}Hz duplicates an extracted mode (<{dup_freq_hz}Hz)")
            break

        center_freqs.append(cf)
        modes.append(u_L)

        print(f"  Step {step+1}: center_freq = {cf:6.1f} Hz, "
              f"residual_energy = {residual_energy:.6f} ({rel_energy:.2%})")

        # 能量停止
        if rel_energy < epsilon:
            print(f"  -> Stopping: residual energy below threshold ({epsilon})")
            break

        # 平台停止: 残差能量几乎不再下降
        energy_drop = (prev_res_energy - residual_energy) / (prev_res_energy + 1e-8)
        if step > 0 and energy_drop < plateau_ratio:
            print(f"  -> Stopping: residual energy plateaued (drop {energy_drop:.2%} < {plateau_ratio:.0%})")
            break
        prev_res_energy = residual_energy

        # 残差直接喂下一步 (residual = input - mode), 全局重构精确成立
        current_input = f_r

    final_residual = f_r.squeeze().cpu().numpy()
    return modes, center_freqs, final_residual


# ============================================================================
# 可视化
# ============================================================================

def plot_results(signal, modes, center_freqs, residual, gt_modes, sr, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    T = len(signal)
    t = np.arange(T) / sr
    n_modes = len(modes)
    n_gt = len(gt_modes)

    # --- 图1：合成信号（时域|频域）---
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # 时域
    axes[0].plot(t, signal, color='gray', linewidth=0.6)
    axes[0].set_title('Mixed Signal (Time Domain)', fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # 频域
    win = np.hanning(len(signal))
    mag = np.abs(np.fft.rfft(signal * win))
    freqs_fft = np.fft.rfftfreq(len(signal), 1/sr)
    axes[1].semilogy(freqs_fft, mag, color='gray', linewidth=0.6)
    axes[1].set_title('Mixed Signal (Frequency Domain)', fontweight='bold')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (log)')
    axes[1].set_xlim(0, sr/2)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '01_mixed_signal.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: 01_mixed_signal.png")

    # --- 图1b：GT子信号（每个子信号一行：时域|频域）---
    fig, axes = plt.subplots(n_gt, 2, figsize=(12, 3 * n_gt))
    if n_gt == 1:
        axes = axes.reshape(1, 2)

    for i, gm in enumerate(gt_modes):
        # 时域（左列）
        axes[i, 0].plot(t, gm, color=f'C{i}', linewidth=0.6)
        axes[i, 0].set_title(f'GT Mode {i+1} (Time Domain)', fontweight='bold')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True, alpha=0.3)

        # 频域（右列）
        win = np.hanning(len(gm))
        mag = np.abs(np.fft.rfft(gm * win))
        freqs_fft = np.fft.rfftfreq(len(gm), 1/sr)
        axes[i, 1].semilogy(freqs_fft, mag, color=f'C{i}', linewidth=0.6)
        
        # 计算中心频率
        power = mag ** 2
        center_freq = np.sum(freqs_fft * power) / np.sum(power)
        axes[i, 1].set_title(f'GT Mode {i+1} (Freq Domain) - CF: {center_freq:.1f} Hz', fontweight='bold')
        axes[i, 1].set_xlabel('Frequency (Hz)')
        axes[i, 1].set_ylabel('Magnitude (log)')
        axes[i, 1].set_xlim(0, sr/2)
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '01b_gt_modes.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: 01b_gt_modes.png")

    # --- 图2：2×2对比图 ---
    # 确定行数：取最大值
    max_rows = max(n_gt, n_modes)
    
    fig, axes = plt.subplots(max_rows, 2, figsize=(12, 3 * max_rows))
    if max_rows == 1:
        axes = axes.reshape(1, 2)

    # 左列：GT子信号（时域）
    for i in range(max_rows):
        if i < n_gt:
            axes[i, 0].plot(t, gt_modes[i], color=f'C{i}', linewidth=0.6)
            axes[i, 0].set_title(f'GT Mode {i+1} (Time)', fontweight='bold')
        else:
            axes[i, 0].set_title(f'GT Mode {i+1} (Time) - N/A', fontweight='bold')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True, alpha=0.3)

    # 右列：分解模态（时域）
    for i in range(max_rows):
        if i < n_modes:
            axes[i, 1].plot(t, modes[i], color=f'C{i}', linewidth=0.6)
            axes[i, 1].set_title(f'Extracted Mode {i+1} (Time) - CF: {center_freqs[i]:.1f} Hz', fontweight='bold')
        else:
            axes[i, 1].set_title(f'Extracted Mode {i+1} (Time) - N/A', fontweight='bold')
        axes[i, 1].set_ylabel('Amplitude')
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '02_time_domain_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: 02_time_domain_comparison.png")

    # --- 图3：2×2对比图（频域）---
    fig, axes = plt.subplots(max_rows, 2, figsize=(12, 3 * max_rows))
    if max_rows == 1:
        axes = axes.reshape(1, 2)

    # 左列：GT子信号（频域）
    for i in range(max_rows):
        if i < n_gt:
            win = np.hanning(len(gt_modes[i]))
            mag = np.abs(np.fft.rfft(gt_modes[i] * win))
            freqs_fft = np.fft.rfftfreq(len(gt_modes[i]), 1/sr)
            axes[i, 0].semilogy(freqs_fft, mag, color=f'C{i}', linewidth=0.6)
            
            # 计算中心频率
            power = mag ** 2
            center_freq = np.sum(freqs_fft * power) / np.sum(power)
            axes[i, 0].set_title(f'GT Mode {i+1} (Freq) - CF: {center_freq:.1f} Hz', fontweight='bold')
        else:
            axes[i, 0].set_title(f'GT Mode {i+1} (Freq) - N/A', fontweight='bold')
        axes[i, 0].set_xlabel('Frequency (Hz)')
        axes[i, 0].set_ylabel('Magnitude (log)')
        axes[i, 0].set_xlim(0, sr/2)
        axes[i, 0].grid(True, alpha=0.3)

    # 右列：分解模态（频域）
    for i in range(max_rows):
        if i < n_modes:
            win = np.hanning(len(modes[i]))
            mag = np.abs(np.fft.rfft(modes[i] * win))
            freqs_fft = np.fft.rfftfreq(len(modes[i]), 1/sr)
            axes[i, 1].semilogy(freqs_fft, mag, color=f'C{i}', linewidth=0.6)
            axes[i, 1].set_title(f'Extracted Mode {i+1} (Freq) - CF: {center_freqs[i]:.1f} Hz', fontweight='bold')
        else:
            axes[i, 1].set_title(f'Extracted Mode {i+1} (Freq) - N/A', fontweight='bold')
        axes[i, 1].set_xlabel('Frequency (Hz)')
        axes[i, 1].set_ylabel('Magnitude (log)')
        axes[i, 1].set_xlim(0, sr/2)
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '03_freq_domain_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: 03_freq_domain_comparison.png")

    # --- 图3b：分解后的模态（每个模态一行：时域|频域）---
    fig, axes = plt.subplots(n_modes + 1, 2, figsize=(12, 3 * (n_modes + 1)))

    for i, (mode, cf) in enumerate(zip(modes, center_freqs)):
        # 时域（左列）
        axes[i, 0].plot(t, mode, color=f'C{i}', linewidth=0.6)
        axes[i, 0].set_title(f'Extracted Mode {i+1} (Time Domain)', fontweight='bold')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True, alpha=0.3)

        # 频域（右列）
        win = np.hanning(len(mode))
        mag = np.abs(np.fft.rfft(mode * win))
        freqs_fft = np.fft.rfftfreq(len(mode), 1/sr)
        axes[i, 1].semilogy(freqs_fft, mag, color=f'C{i}', linewidth=0.6)
        axes[i, 1].set_title(f'Extracted Mode {i+1} (Freq Domain) - CF: {cf:.1f} Hz', fontweight='bold')
        axes[i, 1].set_xlabel('Frequency (Hz)')
        axes[i, 1].set_ylabel('Magnitude (log)')
        axes[i, 1].set_xlim(0, sr/2)
        axes[i, 1].grid(True, alpha=0.3)

    # 残差（最后一行）
    # 时域（左列）
    axes[n_modes, 0].plot(t, residual, color='darkred', linewidth=0.6)
    axes[n_modes, 0].set_title('Residual (Time Domain)', fontweight='bold')
    axes[n_modes, 0].set_ylabel('Amplitude')
    axes[n_modes, 0].grid(True, alpha=0.3)

    # 频域（右列）
    win = np.hanning(len(residual))
    res_mag = np.abs(np.fft.rfft(residual * win))
    freqs_fft = np.fft.rfftfreq(len(residual), 1/sr)
    axes[n_modes, 1].semilogy(freqs_fft, res_mag, color='darkred', linewidth=0.6)
    axes[n_modes, 1].set_title('Residual (Freq Domain)', fontweight='bold')
    axes[n_modes, 1].set_xlabel('Frequency (Hz)')
    axes[n_modes, 1].set_ylabel('Magnitude (log)')
    axes[n_modes, 1].set_xlim(0, sr/2)
    axes[n_modes, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '03b_extracted_modes.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: 03b_extracted_modes.png")

    # --- 图4：残差（时域|频域）---
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # 时域
    axes[0].plot(t, residual, color='darkred', linewidth=0.6)
    axes[0].set_title('Residual (Time Domain)', fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # 频域
    win = np.hanning(len(residual))
    res_mag = np.abs(np.fft.rfft(residual * win))
    freqs_fft = np.fft.rfftfreq(len(residual), 1/sr)
    axes[1].semilogy(freqs_fft, res_mag, color='darkred', linewidth=0.6)
    axes[1].set_title('Residual (Frequency Domain)', fontweight='bold')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (log)')
    axes[1].set_xlim(0, sr/2)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '04_residual.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: 04_residual.png")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Neural SVMD — Demo / Inference (synthetic only)')

    parser.add_argument('--model_path', type=str, default='../exp/vmd_v2/best_model.pth.tar',
                        help='训练好的模型路径')
    parser.add_argument('--out_dir', type=str, default='../exp/vmd_v2/show_best_model',
                        help='输出目录')
    parser.add_argument('--sample_rate', type=int, default=8000, help='采样率')
    parser.add_argument('--duration', type=float, default=2.0, help='合成信号时长 (秒)')
    parser.add_argument('--use_cuda', type=int, default=0, help='使用 GPU')
    parser.add_argument('--max_steps', type=int, default=10, help='最大递归步数')
    parser.add_argument('--epsilon', type=float, default=0.005,
                        help='残差相对能量阈值 (停止条件)')

    return parser.parse_args()


def main():
    args = parse_args()

    # 加载模型
    model, device = load_model(args.model_path, use_cuda=args.use_cuda)

    # 生成合成测试信号
    signal, gt_modes, sr = make_test_signal(sr=args.sample_rate, duration=args.duration)
    global SAMPLE_RATE
    SAMPLE_RATE = sr
    print(f"Generated synthetic signal: {len(signal)} samples, {len(gt_modes)} modes")

    print(f"\nDecomposing signal ({len(signal)/args.sample_rate:.2f}s)...")
    modes, center_freqs, residual = decompose(
        model, signal, device, max_steps=args.max_steps, epsilon=args.epsilon
    )

    print(f"\nResult: {len(modes)} modes extracted")
    print(f"Center frequencies (Hz): {[f'{cf:.1f}' for cf in center_freqs]}")

    # 可视化
    print("\nGenerating plots...")
    plot_results(signal, modes, center_freqs, residual, gt_modes, args.sample_rate, args.out_dir)

    print(f"\nDone. Output: {os.path.abspath(args.out_dir)}")
    print(f"  Modes extracted: {len(modes)}")


if __name__ == '__main__':
    main()