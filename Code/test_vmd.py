# Neural SVMD — test on 10 synthetic signals with visualization
#
# Usage:
#   python test_vmd.py --checkpoint ../exp/vmd/best_model.pth.tar
#
# Generates 10 test signals (3–10 modes), decomposes each, and saves:
#   {out_dir}/{name}_original.png  — original mixture (time + spectrum)
#   {out_dir}/{name}_modes.png     — extracted modes (time + spectrum)

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
DEFAULT_CHECKPOINT = '../exp/vmd/best_model.pth.tar'
DEFAULT_OUT_DIR = '../exp/vmd/test_output'
RECURSION_STEPS = 20  # max modes = 10 + 2 buffer


# ============================================================================
# 10 test signal configurations
# ============================================================================

def _make_mode(t, fc, amp, mod_type, mod_depth, seed):
    """Generate a single mode component."""
    rng = np.random.RandomState(seed)
    if mod_type == 'AM':
        am_freq = rng.uniform(3, 10)
        mode = amp * np.sin(2 * np.pi * fc * t) * (1.0 + mod_depth * np.sin(2 * np.pi * am_freq * t))
    elif mod_type == 'FM':
        fm_freq = rng.uniform(4, 12)
        mode = amp * np.sin(2 * np.pi * fc * t + mod_depth * 5 * np.cos(2 * np.pi * fm_freq * t))
    else:  # CW — pure sine
        mode = amp * np.sin(2 * np.pi * fc * t)
    return mode.astype(np.float32)


# Each config: list of (center_freq_Hz, amplitude, modulation_type, modulation_depth, random_seed)
TEST_SIGNAL_CONFIGS = [
    # 1) 3 modes — well-separated
    ("test_01_3modes_clean", [
        (300,  1.0, 'AM', 0.3, 10),
        (1200, 0.8, 'FM', 0.4, 20),
        (3000, 0.6, 'CW', 0.0, 30),
    ]),
    # 2) 4 modes — mid-range
    ("test_02_4modes_mid", [
        (200,  1.0, 'AM', 0.2, 11),
        (800,  0.7, 'CW', 0.0, 21),
        (1600, 0.6, 'FM', 0.3, 31),
        (2800, 0.5, 'AM', 0.4, 41),
    ]),
    # 3) 5 modes — evenly spaced
    ("test_03_5modes_even", [
        (250,  1.0, 'CW', 0.0, 12),
        (750,  0.9, 'AM', 0.3, 22),
        (1250, 0.8, 'FM', 0.3, 32),
        (1750, 0.7, 'AM', 0.2, 42),
        (2250, 0.6, 'CW', 0.0, 52),
    ]),
    # 4) 6 modes — two are close in frequency (600 & 650 Hz)
    ("test_04_6modes_close", [
        (300,  0.9, 'AM', 0.3, 13),
        (600,  0.8, 'FM', 0.3, 23),
        (650,  0.7, 'CW', 0.0, 33),  # close to 600 Hz
        (1500, 0.6, 'AM', 0.4, 43),
        (2500, 0.5, 'FM', 0.2, 53),
        (3500, 0.4, 'CW', 0.0, 63),
    ]),
    # 5) 7 modes — wide frequency range
    ("test_05_7modes_wide", [
        (150,  1.0, 'CW', 0.0, 14),
        (500,  0.9, 'AM', 0.3, 24),
        (900,  0.8, 'FM', 0.4, 34),
        (1400, 0.7, 'AM', 0.2, 44),
        (2000, 0.6, 'CW', 0.0, 54),
        (2700, 0.5, 'FM', 0.3, 64),
        (3500, 0.4, 'AM', 0.3, 74),
    ]),
    # 6) 8 modes — relatively dense
    ("test_06_8modes_dense", [
        (200,  0.9, 'AM', 0.3, 15),
        (500,  0.8, 'CW', 0.0, 25),
        (800,  0.7, 'FM', 0.3, 35),
        (1100, 0.7, 'AM', 0.2, 45),
        (1500, 0.6, 'CW', 0.0, 55),
        (2000, 0.5, 'FM', 0.4, 65),
        (2600, 0.5, 'AM', 0.3, 75),
        (3300, 0.4, 'CW', 0.0, 85),
    ]),
    # 7) 9 modes — full spectrum coverage
    ("test_07_9modes_full", [
        (120,  1.0, 'CW', 0.0, 16),
        (400,  0.9, 'AM', 0.3, 26),
        (700,  0.8, 'FM', 0.3, 36),
        (1000, 0.8, 'AM', 0.2, 46),
        (1400, 0.7, 'CW', 0.0, 56),
        (1800, 0.6, 'FM', 0.4, 66),
        (2300, 0.5, 'AM', 0.3, 76),
        (2900, 0.5, 'CW', 0.0, 86),
        (3600, 0.4, 'FM', 0.3, 96),
    ]),
    # 8) 10 modes — maximum density
    ("test_08_10modes_max", [
        (150,  0.8, 'AM', 0.2, 17),
        (400,  0.7, 'CW', 0.0, 27),
        (650,  0.7, 'FM', 0.3, 37),
        (900,  0.6, 'AM', 0.3, 47),
        (1200, 0.6, 'CW', 0.0, 57),
        (1600, 0.5, 'FM', 0.3, 67),
        (2000, 0.5, 'AM', 0.2, 77),
        (2500, 0.4, 'CW', 0.0, 87),
        (3000, 0.4, 'FM', 0.3, 97),
        (3600, 0.3, 'AM', 0.3, 107),
    ]),
    # 9) 4 modes — low-frequency cluster
    ("test_09_4modes_low", [
        (200, 1.0, 'FM', 0.4, 18),
        (350, 0.9, 'AM', 0.3, 28),
        (500, 0.8, 'CW', 0.0, 38),
        (700, 0.7, 'AM', 0.3, 48),
    ]),
    # 10) 5 modes — mixed modulation
    ("test_10_5modes_mixed", [
        (400,  1.0, 'FM', 0.5, 19),
        (1000, 0.8, 'AM', 0.4, 29),
        (1800, 0.7, 'CW', 0.0, 39),
        (2400, 0.6, 'AM', 0.3, 49),
        (3200, 0.5, 'FM', 0.3, 59),
    ]),
]


# ============================================================================
# Helpers
# ============================================================================

def build_test_signal(config, duration=1.0, sr=SAMPLE_RATE):
    """Build mixture + ground-truth components from a config list."""
    T = int(sr * duration)
    t = np.linspace(0, duration, T, endpoint=False)
    components = []
    mixture = np.zeros(T, dtype=np.float32)
    for fc, amp, mod_type, mod_depth, seed in config:
        mode = _make_mode(t, fc, amp, mod_type, mod_depth, seed)
        components.append(mode)
        mixture += mode
    # Add light noise
    mixture += 0.01 * np.random.RandomState(999).randn(T).astype(np.float32)
    return mixture.astype(np.float32), components, sr


def load_model(checkpoint_path, device):
    """Load trained NeuralSVMD model."""
    model = NeuralSVMD(n_fft=512, hop_length=128, hidden_dim=64, sample_rate=SAMPLE_RATE)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


@torch.no_grad()
def decompose(model, signal, device, max_steps=RECURSION_STEPS, epsilon=0.005,
              plateau_ratio=0.02, dup_freq_hz=50.0):
    """Recursively decompose signal into modes.

    残差直接喂下一步 (residual = input - mode), 全局重构精确成立.

    三重停止判据 (任一满足即停, 防止真实模态提完后继续产出重复/混叠的伪模态):
      1. 能量停止 : 残差能量 / 原能量 < epsilon          (残差已足够小)
      2. 平台停止 : 残差能量相比上一步降幅 < plateau_ratio (提不出新东西了)
      3. 重复停止 : 新模态中心频率与「任一」已提取模态接近 < dup_freq_hz
                    (开始重复提同一频带 = 混叠)
    """
    signal_tensor = torch.from_numpy(signal).float().unsqueeze(0).to(device)
    original_energy = torch.mean(signal_tensor ** 2).item()

    modes = []
    center_freqs = []
    current_input = signal_tensor
    prev_res_energy = original_energy

    for step in range(max_steps):
        output = model(current_input, step)
        u_L = output["mode"][0, :].cpu().numpy()
        f_r = output["residual"]      # [1, T]

        cf = compute_center_freq(output["mode"], sample_rate=SAMPLE_RATE).item()

        residual_energy = compute_residual_energy(f_r).item()
        rel_energy = residual_energy / (original_energy + 1e-8)

        # ---- 重复停止: 新模态与任一历史模态中心频率过近 → 视为混叠, 丢弃并停 ----
        if center_freqs and min(abs(cf - c) for c in center_freqs) < dup_freq_hz:
            break

        center_freqs.append(cf)
        modes.append(u_L)

        # ---- 能量停止: 残差已足够小 ----
        if rel_energy < epsilon:
            break

        # ---- 平台停止: 残差能量几乎不再下降 → 提不出新东西 ----
        energy_drop = (prev_res_energy - residual_energy) / (prev_res_energy + 1e-8)
        if step > 0 and energy_drop < plateau_ratio:
            break
        prev_res_energy = residual_energy

        current_input = f_r

    final_residual = f_r.squeeze().cpu().numpy()
    return modes, center_freqs, final_residual


# ============================================================================
# Plotting
# ============================================================================

def plot_original(name, mixture, components, gt_configs, sr, out_dir):
    """
    Figure 1: Original signal.
      Left column  = time domain (row 0 = mixture, rows 1..N = ground-truth components)
      Right column = frequency domain (same layout)
    """
    T = len(mixture)
    n_components = len(components)
    n_rows = 1 + n_components  # mixture + components

    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 2.2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    t_axis = np.arange(T) / sr
    freqs = np.fft.rfftfreq(T, 1.0 / sr)

    def _plot_row(ax_t, ax_f, data, title, color):
        # Time domain
        ax_t.plot(t_axis, data, color=color, linewidth=0.5)
        ax_t.set_ylabel('Amplitude')
        ax_t.set_title(title, fontsize=9)
        ax_t.grid(True, alpha=0.3)
        ax_t.set_xlim(0, t_axis[-1])
        # Freq domain
        spec = np.abs(np.fft.rfft(data * np.hanning(T)))
        ax_f.plot(freqs, spec, color=color, linewidth=0.5)
        ax_f.set_ylabel('Magnitude')
        ax_f.set_title(title, fontsize=9)
        ax_f.grid(True, alpha=0.3)
        ax_f.set_xlim(0, sr / 2)

    # Row 0: mixture
    _plot_row(axes[0, 0], axes[0, 1], mixture, f'{name} — Mixture ({n_components} modes)', '#1f77b4')

    # Rows 1..N: ground-truth components
    comp_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
                   '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4']
    for j, comp in enumerate(components):
        fc, _, mod_type, _, _ = gt_configs[j]
        _plot_row(axes[1 + j, 0], axes[1 + j, 1], comp,
                  f'GT Mode {j+1} — {fc} Hz ({mod_type})',
                  comp_colors[j % len(comp_colors)])

    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Frequency (Hz)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_original.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}_original.png")


def plot_modes(name, modes_data, sr, out_dir, residual=None):
    """
    Figure 2: Decomposed modes.
      Left column  = time domain
      Right column = frequency domain
      One row per extracted mode; 若传入 residual, 末尾追加一行最终残差.
    """
    n_modes = len(modes_data)
    n_rows = n_modes + (1 if residual is not None else 0)
    T = len(modes_data[0][0])

    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 2.2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    t_axis = np.arange(T) / sr
    freqs = np.fft.rfftfreq(T, 1.0 / sr)
    colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#e377c2', '#7f7f7f',
              '#d62728', '#8c564b', '#bcbd22', '#17becf', '#1f77b4',
              '#1a9850', '#fee090']

    for i, (mode_np, cf) in enumerate(modes_data):
        ax_t, ax_f = axes[i, 0], axes[i, 1]
        color = colors[i % len(colors)]

        # Time domain
        ax_t.plot(t_axis, mode_np, color=color, linewidth=0.5)
        ax_t.set_ylabel('Amplitude')
        ax_t.set_title(f'Mode {i+1} — CF = {cf:.0f} Hz', fontsize=9)
        ax_t.grid(True, alpha=0.3)
        ax_t.set_xlim(0, t_axis[-1])

        # Frequency domain
        spec = np.abs(np.fft.rfft(mode_np * np.hanning(T)))
        ax_f.plot(freqs, spec, color=color, linewidth=0.5)
        ax_f.set_ylabel('Magnitude')
        ax_f.set_title(f'Mode {i+1} — CF = {cf:.0f} Hz', fontsize=9)
        ax_f.grid(True, alpha=0.3)
        ax_f.set_xlim(0, sr / 2)

    # ---- 末行: 最终残差 (区别于分解模态, 用黑色显示) ----
    if residual is not None:
        ax_t, ax_f = axes[n_modes, 0], axes[n_modes, 1]
        res_energy = float(np.mean(residual ** 2))
        ax_t.plot(t_axis, residual, color='black', linewidth=0.5)
        ax_t.set_ylabel('Amplitude')
        ax_t.set_title(f'Residual — energy = {res_energy:.2e}', fontsize=9)
        ax_t.grid(True, alpha=0.3)
        ax_t.set_xlim(0, t_axis[-1])

        spec = np.abs(np.fft.rfft(residual * np.hanning(T)))
        ax_f.plot(freqs, spec, color='black', linewidth=0.5)
        ax_f.set_ylabel('Magnitude')
        ax_f.set_title('Residual (spectrum)', fontsize=9)
        ax_f.grid(True, alpha=0.3)
        ax_f.set_xlim(0, sr / 2)

    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Frequency (Hz)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_modes.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}_modes.png")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Neural SVMD — Test on 10 synthetic signals')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, help='Model checkpoint path')
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR, help='Output directory for plots')
    parser.add_argument('--duration', type=float, default=1.0, help='Signal duration in seconds')
    parser.add_argument('--use_cuda', type=int, default=0, help='Use GPU')
    args = parser.parse_args()

    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, device)

    # Test each signal
    print(f"\n{'='*70}")
    print(f"Testing {len(TEST_SIGNAL_CONFIGS)} synthetic signals")
    print(f"{'='*70}")

    for name, config in TEST_SIGNAL_CONFIGS:
        n_gt = len(config)
        print(f"\n{'—'*50}")
        print(f"Test: {name} ({n_gt} ground-truth modes)")
        print(f"  GT frequencies: {[f'{fc}Hz' for fc, _, _, _, _ in config]}")

        # Build signal
        mixture, gt_components, sr = build_test_signal(config, duration=args.duration)

        # Decompose
        modes, center_freqs, residual = decompose(model, mixture, device)

        print(f"  Extracted {len(modes)} modes")
        print(f"  CFs: {[f'{cf:.0f}Hz' for cf in center_freqs]}")

        # Compute reconstruction MSE
        recon = np.sum(modes, axis=0) + residual
        mse = np.mean((mixture - recon) ** 2)
        print(f"  Recon MSE: {mse:.6f}")

        # Plot
        plot_original(name, mixture, gt_components, config, sr, args.out_dir)
        modes_with_cf = list(zip(modes, center_freqs))
        plot_modes(name, modes_with_cf, sr, args.out_dir, residual=residual)

    print(f"\n{'='*70}")
    print(f"Done. Results saved to: {os.path.abspath(args.out_dir)}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()