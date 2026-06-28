#!/usr/bin/env python3
"""
Neural SVMD — 轻量级训练 + 测试 Demo

快速训练一个小模型（~5-10分钟 CPU），然后对合成信号进行模态分解并可视化。
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import NeuralSVMD
from losses import NeuralSVMDCriterion, compute_center_freq


# ══════════════════════════════════════════════════════════════════════════════
# 轻量级合成信号生成
# ══════════════════════════════════════════════════════════════════════════════

def generate_light_signal(batch_size, signal_length, sample_rate=8000,
                           min_modes=2, max_modes=4, noise_std=0.02,
                           min_spacing=150):
    """生成简单的多分量 AM-FM 信号。"""
    t = torch.arange(signal_length, dtype=torch.float32) / sample_rate
    signals = []

    for _ in range(batch_size):
        k = np.random.randint(min_modes, max_modes + 1)
        # 在 [150, sr/2-150] Hz 范围内随机放置中心频率
        f_min, f_max = 150, sample_rate // 2 - 150
        available = f_max - f_min - (k - 1) * min_spacing
        if available <= 0:
            centers = sorted(np.random.uniform(f_min, f_max, k))
        else:
            offsets = np.sort(np.random.uniform(0, available, k))
            centers = [f_min + offsets[i] + i * min_spacing for i in range(k)]

        mixture = torch.zeros(signal_length)
        for fc in centers:
            am = 1.0 + 0.2 * torch.sin(2 * np.pi * np.random.uniform(2, 6) * t)
            fm = torch.sin(2 * np.pi * fc * t +
                           np.random.uniform(0.5, 3) * torch.cos(2 * np.pi * np.random.uniform(3, 10) * t))
            mixture += np.random.uniform(0.5, 1.5) * am * fm

        mixture += noise_std * torch.randn(signal_length)
        signals.append(mixture)

    return torch.stack(signals)


# ══════════════════════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════════════════════

def train_light(args):
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = NeuralSVMD(n_fft=args.n_fft, hop_length=args.hop_length,
                        hidden_dim=args.hidden_dim, sample_rate=args.sample_rate)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    criterion = NeuralSVMDCriterion(
        alpha=args.alpha_bandwidth,
        eps=1e-8,
        momentum=0.95,
        w1=args.weight_bandwidth,
        w2=args.gamma_residual,
        w3=args.delta_history,
        sample_rate=args.sample_rate,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.98), eps=1e-9,
                                  weight_decay=args.weight_decay)

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')

    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_j1 = epoch_j2 = epoch_j3 = 0.0

        for batch_idx in range(args.batches_per_epoch):
            batch = generate_light_signal(
                args.batch_size, args.signal_length, args.sample_rate,
                args.min_modes, args.max_modes, args.noise_std
            ).to(device)

            total_loss = 0.0
            current_input = batch
            history = []
            orig_energy = torch.mean(batch ** 2, dim=-1)
            actual_steps = 0

            for step in range(args.recursion_steps):
                output = model(current_input, step)
                u_L = output["mode"]
                f_r = output["residual"]

                result = criterion(
                    output["mode_spec"],
                    output["residual_spec"],
                    output["signal_spec"],
                    history,
                )
                step_loss = result["loss"]
                J1 = result["j1"]
                J2 = result["j2"]
                J3 = result["j3"]

                total_loss += step_loss
                actual_steps += 1
                history.append({
                    "omega": result["centroid"].mean().detach(),
                    "mode_spec": output["mode_spec"].detach(),
                })
                criterion.update_omega()
                current_input = f_r

                epoch_j1 += J1.item(); epoch_j2 += J2.item()
                epoch_j3 += J3.item()

                # 自适应停止
                res_energy = torch.mean(f_r ** 2, dim=-1)
                if (res_energy < args.stop_eps * orig_energy).all():
                    break

            total_loss = total_loss / actual_steps

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / args.batches_per_epoch
        n = args.batches_per_epoch * args.recursion_steps

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss {avg_loss:.6f} | "
                  f"J1={epoch_j1/n:.4f} J2={epoch_j2/n:.4f} "
                  f"J3={epoch_j3/n:.4f} | "
                  f"omega={criterion.omega.item()*args.sample_rate:.1f}Hz")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'criterion_state': criterion.state_dict(),
            }, os.path.join(args.save_dir, 'best_model.pth.tar'))

    elapsed = time.time() - t_start
    print(f"\nTraining done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Best loss: {best_loss:.6f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 测试信号
# ══════════════════════════════════════════════════════════════════════════════

def make_test_signal(sr=8000, duration=1.0):
    """生成一个含 5 个模态的已知测试信号。"""
    T = int(sr * duration)
    t = np.arange(T, dtype=np.float32) / sr
    np.random.seed(42)

    modes = []
    configs = [
        (200,  0.2, 1.0,  'AM'),   # 低频 AM
        (600,  0.0, 0.8,  'CW'),   # 纯正弦
        (1200, 0.0, 0.6,  'FM'),   # FM
        (2000, 0.3, 0.5,  'AM'),   # 中高频 AM
        (3000, 0.0, 0.3,  'CW'),   # 高频纯正弦
    ]

    for fc, mod_depth, amp, mod_type in configs:
        if mod_type == 'AM':
            mode = amp * np.sin(2 * np.pi * fc * t) * \
                   (1 + mod_depth * np.sin(2 * np.pi * np.random.uniform(3, 8) * t))
        elif mod_type == 'FM':
            mode = amp * np.sin(2 * np.pi * fc * t +
                                mod_depth * 5 * np.cos(2 * np.pi * np.random.uniform(3, 10) * t))
        else:
            mode = amp * np.sin(2 * np.pi * fc * t)
        modes.append(mode.astype(np.float32))

    mixture = np.sum(modes, axis=0) + 0.02 * np.random.randn(T).astype(np.float32)
    return mixture.astype(np.float32), modes, sr


# ══════════════════════════════════════════════════════════════════════════════
# 递归分解
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def decompose(model, signal, device, max_steps=8, epsilon=0.005, sr=8000):
    signal_t = torch.from_numpy(signal).float().unsqueeze(0).to(device)
    orig_energy = torch.mean(signal_t ** 2).item()

    modes, freqs = [], []
    current_input = signal_t

    for step in range(max_steps):
        output = model(current_input, step)
        u_L = output["mode"][0, :].cpu().numpy()
        f_r = output["residual"]  # [1, T]

        cf = compute_center_freq(output["mode"], sample_rate=sr).item()
        freqs.append(cf)
        modes.append(u_L)

        res_energy = torch.mean(f_r ** 2).item()
        rel = res_energy / (orig_energy + 1e-8)
        print(f"  Step {step+1}: w_c = {cf:7.1f} Hz, residual = {res_energy:.6f} ({rel:.2%})")

        if rel < epsilon:
            print(f"  -> 停止: 残差能量低于阈值")
            break
        # 残差直接喂下一步 (residual = input - mode), 全局重构精确成立
        current_input = f_r

    return modes, freqs, f_r.squeeze().cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(signal, modes, freqs, residual, gt_modes, sr, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    T = len(signal)
    t = np.arange(T) / sr
    n_modes = len(modes)
    n_gt = len(gt_modes)

    # --- 图1：Ground Truth 子信号（每个子信号一行：时域|频域）---
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
        axes[i, 1].set_title(f'GT Mode {i+1} (Freq Domain) - Center: {center_freq:.1f} Hz', fontweight='bold')
        axes[i, 1].set_xlabel('Frequency (Hz)')
        axes[i, 1].set_ylabel('Magnitude (log)')
        axes[i, 1].set_xlim(0, sr/2)
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '01_ground_truth.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: 01_ground_truth.png")

    # --- 图2：分解后的模态（每个模态一行：时域|频域）---
    fig, axes = plt.subplots(n_modes + 1, 2, figsize=(12, 3 * (n_modes + 1)))

    for i, (mode, cf) in enumerate(zip(modes, freqs)):
        # 时域（左列）
        axes[i, 0].plot(t, mode, color=f'C{i}', linewidth=0.6)
        axes[i, 0].set_title(f'Mode {i+1} (Time Domain)', fontweight='bold')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True, alpha=0.3)

        # 频域（右列）
        win = np.hanning(len(mode))
        mag = np.abs(np.fft.rfft(mode * win))
        freqs_fft = np.fft.rfftfreq(len(mode), 1/sr)
        axes[i, 1].semilogy(freqs_fft, mag, color=f'C{i}', linewidth=0.6)
        axes[i, 1].set_title(f'Mode {i+1} (Freq Domain) - Center: {cf:.1f} Hz', fontweight='bold')
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
    fig.savefig(os.path.join(out_dir, '02_decomposed_modes.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: 02_decomposed_modes.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Neural SVMD — Lightweight Demo')
    # Data
    parser.add_argument('--signal_length', type=int, default=2048)
    parser.add_argument('--sample_rate', type=int, default=8000)
    parser.add_argument('--min_modes', type=int, default=2)
    parser.add_argument('--max_modes', type=int, default=4)
    parser.add_argument('--noise_std', type=float, default=0.02)
    # Model
    parser.add_argument('--n_fft', type=int, default=256)
    parser.add_argument('--hop_length', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=32)
    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batches_per_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--max_norm', type=float, default=5.0)
    parser.add_argument('--recursion_steps', type=int, default=5)
    parser.add_argument('--stop_eps', type=float, default=0.01)
    # Loss weights (J1/J2/J3, 归一化频率轴)
    parser.add_argument('--alpha_bandwidth', type=float, default=50.0,
                        help='beta filter sharpness (paper alpha)')
    parser.add_argument('--weight_bandwidth', type=float, default=20.0,
                        help='w1: J1 bandwidth compactness weight')
    parser.add_argument('--gamma_residual', type=float, default=1.0,
                        help='w2: J2 residual exclusion weight')
    parser.add_argument('--delta_history', type=float, default=2.0,
                        help='w3: J3 history orthogonality weight')
    # System
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='../exp/vmd_light')
    parser.add_argument('--out_dir', type=str, default='../exp/vmd_light/demo')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training, load existing checkpoint')
    parser.add_argument('--checkpoint', type=str, default='')

    args = parser.parse_args()

    print("=" * 70)
    print("Neural SVMD — Lightweight Training + Inference Demo")
    print("=" * 70)
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("=" * 70)

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # ── Train ──────────────────────────────────────────────────────────────
    ckpt_path = args.checkpoint or os.path.join(args.save_dir, 'best_model.pth.tar')

    if args.skip_train and os.path.exists(ckpt_path):
        print(f"\nSkipping training, loading: {ckpt_path}")
        model = NeuralSVMD(n_fft=args.n_fft, hop_length=args.hop_length,
                            hidden_dim=args.hidden_dim,
                            sample_rate=args.sample_rate).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("\n" + "=" * 70)
        print("Phase 1: Training")
        print("=" * 70)
        model = train_light(args)
        print(f"\nCheckpoint saved to: {ckpt_path}")

    # ── Test ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 2: Inference on Test Signal")
    print("=" * 70)

    model.eval()
    signal, gt_modes, sr = make_test_signal(sr=args.sample_rate, duration=1.0)
    print(f"Test signal: {len(signal)} samples ({len(signal)/sr:.2f}s), "
          f"{len(gt_modes)} ground-truth modes")

    print("\nDecomposing...")
    modes, freqs, residual = decompose(model, signal, device,
                                        max_steps=args.recursion_steps,
                                        epsilon=args.stop_eps, sr=sr)

    print(f"\nExtracted {len(modes)} modes:")
    for i, cf in enumerate(freqs):
        print(f"  Mode {i+1}: center frequency = {cf:.1f} Hz")

    recon = np.sum(modes, axis=0) + residual
    mse = np.mean((signal - recon) ** 2)
    print(f"\nReconstruction MSE: {mse:.6f}")

    # ── Plot ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 3: Visualization")
    print("=" * 70)
    plot_all(signal, modes, freqs, residual, gt_modes, sr, args.out_dir)

    print(f"\nAll outputs saved to: {os.path.abspath(args.out_dir)}")
    print("Done!")


if __name__ == '__main__':
    main()