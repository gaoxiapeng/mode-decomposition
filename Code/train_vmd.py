# Neural SVMD — unsupervised recursive mode decomposition training
# Frequency-domain architecture with SVMD variational loss

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import NeuralSVMD
from losses import NeuralSVMDCriterion


# ============================================================================
# Synthetic data generation: multi-component AM-FM signals
# ============================================================================

def generate_synthetic_signal(batch_size, signal_length, sample_rate=8000,
                               min_modes=3, max_modes=10, noise_std=0.02,
                               min_spacing=80):
    """
    Generate synthetic multi-component signals for unsupervised training.

    Each signal contains K modes (K in [min_modes, max_modes] randomly chosen).
    Each mode is an AM-FM sine with minimum inter-mode frequency spacing.

    Returns:
        signal: [B, T] mixture
    """
    t = torch.arange(signal_length, dtype=torch.float32) / sample_rate
    t = t.unsqueeze(0)  # [1, T]

    signals = []

    for b in range(batch_size):
        k = np.random.randint(min_modes, max_modes + 1)

        f_min, f_max = 100, sample_rate // 2 - 100
        available = f_max - f_min - (k - 1) * min_spacing
        if available <= 0:
            freq_centers = sorted(np.random.uniform(f_min, f_max, k))
        else:
            offsets = np.sort(np.random.uniform(0, available, k))
            freq_centers = [f_min + offsets[i] + i * min_spacing for i in range(k)]

        modes_b = []
        for fc in freq_centers:
            am = 1.0 + 0.3 * torch.sin(2 * np.pi * np.random.uniform(2, 8) * t)
            fm = torch.sin(2 * np.pi * fc * t +
                           np.random.uniform(1, 4) * torch.cos(2 * np.pi * np.random.uniform(3, 15) * t))
            mode = np.random.uniform(0.5, 1.5) * am * fm
            modes_b.append(mode.squeeze(0))

        mixture = torch.stack(modes_b).sum(dim=0)  # [T]
        mixture = mixture + noise_std * torch.randn(signal_length)
        signals.append(mixture)

    return torch.stack(signals)  # [B, T]


# ============================================================================
# Learning rate scheduler
# ============================================================================

class WarmupDecayOptimizer:
    """Transformer-style warmup + exponential decay."""

    def __init__(self, optimizer, d_model=64, warmup_steps=4000, k=0.2):
        self.optimizer = optimizer
        self.init_lr = d_model ** (-0.5)
        self.warmup_steps = warmup_steps
        self.k = k
        self.step_num = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, epoch):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.k * self.init_lr * min(
                self.step_num ** (-0.5),
                self.step_num * (self.warmup_steps ** (-1.5))
            )
        else:
            lr = 0.0004 * (0.98 ** ((epoch - 1) // 2))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


# ============================================================================
# Training
# ============================================================================

def train(args):
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    # AMP 默认关闭: 频域损失对数值精度敏感, FP32 训练更稳 (防 NaN)
    use_amp = (device.type == 'cuda') and bool(args.use_amp)
    print(f"Using device: {device}")
    print(f"AMP (mixed precision): {'enabled' if use_amp else 'disabled (FP32)'}")

    # ---- Model ----
    model = NeuralSVMD(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        hidden_dim=args.hidden_dim,
        sample_rate=args.sample_rate,
    )
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {param_count:,}")

    # ---- Criterion ----
    criterion = NeuralSVMDCriterion(
        alpha=args.alpha_bandwidth,
        eps=1e-8,
        momentum=0.95,
        w1=args.weight_bandwidth,
        w2=args.gamma_residual,
        w3=args.delta_history,
        sample_rate=args.sample_rate,
        max_steps=args.recursion_steps,
    )
    criterion.to(device)

    # ---- Optimizer ----
    adam = torch.optim.Adam(
        model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=args.lr,
        weight_decay=args.weight_decay
    )
    optimizer = WarmupDecayOptimizer(adam, d_model=64, warmup_steps=args.warmup_steps)

    # ---- Learning Rate Scheduler ----
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )

    # AMP gradient scaler (GPU only)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ---- Save directory ----
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Resume ----
    start_epoch = 0
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        state_dict = ckpt['model_state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'criterion_state' in ckpt:
            # omegas 等 buffer 形状随 max_steps 变化; 旧 checkpoint 形状不符时跳过
            try:
                criterion.load_state_dict(ckpt['criterion_state'])
            except (RuntimeError, ValueError) as e:
                print(f"  [warn] criterion_state 形状不兼容, 跳过加载 (omegas 将从 0 重新学习): {e}")
        start_epoch = ckpt['epoch']
        print(f"Resumed from epoch {start_epoch}")

    # ---- Training loop ----
    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_j1, epoch_j2, epoch_j3 = 0.0, 0.0, 0.0
        n_batches = args.batches_per_epoch

        for batch_idx in range(n_batches):
            # ---- Direct batch generation (no DataLoader overhead) ----
            batch_signals = generate_synthetic_signal(
                args.batch_size, args.signal_length, args.sample_rate,
                args.min_modes, args.max_modes, args.noise_std
            )
            batch_signals = batch_signals.to(device, non_blocking=True)

            total_loss = 0.0
            current_input = batch_signals
            history = []
            original_energy = torch.mean(batch_signals ** 2, dim=-1)  # [B]

            # ---- Recursive mode extraction with adaptive stopping ----
            actual_steps = 0
            for step in range(args.recursion_steps):
                # AMP: model forward in mixed precision, loss in FP32
                if use_amp:
                    with torch.cuda.amp.autocast():
                        output = model(current_input, step)
                else:
                    output = model(current_input, step)

                u_L = output["mode"]
                f_r = output["residual"]

                result = criterion(
                    output["mode_spec"].float(),
                    output["residual_spec"].float(),
                    output["signal_spec"].float(),
                    history,
                    step,
                )
                step_loss = result["loss"]
                J1 = result["j1"]
                J2 = result["j2"]
                J3 = result["j3"]

                total_loss += step_loss
                actual_steps += 1

                # 先更新该 step 的 omegas[step] (仅日志监控), history 用逐样本质心驱动 J3
                criterion.update_omega(step)
                history.append({
                    "omega": result["centroid"].detach(),   # [B] 逐样本质心 (供 J3 的 β_i)
                    "mode_spec": output["mode_spec"].detach(),
                })
                current_input = f_r.float()

                epoch_j1 += J1.item()
                epoch_j2 += J2.item()
                epoch_j3 += J3.item()

                # Adaptive stopping: break if residual energy is low for all samples
                residual_energy = torch.mean(f_r ** 2, dim=-1)  # [B]
                if (residual_energy < args.stop_epsilon * original_energy).all():
                    break

            total_loss = total_loss / actual_steps

            # NaN检测
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"\n{'='*80}")
                print(f"WARNING: NaN/Inf detected at Epoch {epoch+1}, Batch {batch_idx}")
                print(f"  Loss: {total_loss}")
                print(f"  Saving emergency checkpoint and stopping training...")
                print(f"{'='*80}\n")
                
                # 保存紧急checkpoint
                emergency_path = os.path.join(args.save_dir, 'emergency_checkpoint.pth.tar')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                    'criterion_state': criterion.state_dict(),
                    'loss': total_loss,
                }, emergency_path)
                print(f"Emergency checkpoint saved: {emergency_path}")
                return

            # Loss异常检测 (J2/J3 有界后损失应在 O(10) 量级, 阈值收紧到 100)
            if total_loss > 100:
                print(f"\n{'='*80}")
                print(f"WARNING: Loss exploded at Epoch {epoch+1}, Batch {batch_idx}")
                print(f"  Loss: {total_loss:.2e}")
                print(f"  Skipping this batch...")
                print(f"{'='*80}\n")
                continue

            optimizer.zero_grad()

            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer.optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                scaler.step(optimizer.optimizer)
                scaler.update()
                # LR update (scaler already stepped parameters, avoid double-step)
                optimizer.step_num += 1
                if optimizer.step_num <= optimizer.warmup_steps:
                    lr = optimizer.k * optimizer.init_lr * min(
                        optimizer.step_num ** (-0.5),
                        optimizer.step_num * (optimizer.warmup_steps ** (-1.5))
                    )
                else:
                    lr = 0.0004 * (0.98 ** ((epoch - 1) // 2))
                for param_group in optimizer.optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                optimizer.step(epoch)

            epoch_loss += total_loss.item()

            if batch_idx % args.log_interval == 0:
                cf_str = " | ".join(
                    f"omega{i+1}={history[i]['omega'].mean().item()*args.sample_rate:.1f}Hz"
                    for i in range(min(len(history), 3))
                )
                print(f"Epoch {epoch+1:3d} | Batch {batch_idx:4d} | "
                      f"Loss {total_loss.item():.6f} | "
                      f"J1={J1.item():.4f} J2={J2.item():.4f} J3={J3.item():.4f} | "
                      f"CFs: {cf_str}",
                      flush=True)

        # ---- Epoch summary ----
        avg_loss = epoch_loss / n_batches
        elapsed = time.time() - epoch_start

        # 显示前几个 step 的中心频率 (各 step 独立维护)
        n_show = min(criterion.max_steps, 5)
        omega_str = " ".join(
            f"{criterion.omegas[i].item()*args.sample_rate:.0f}"
            for i in range(n_show)
        )

        print(f"{'='*80}")
        print(f"Epoch {epoch+1}/{args.epochs} | Time {elapsed:.1f}s | "
              f"Avg Loss {avg_loss:.6f} | "
              f"J1={epoch_j1/(n_batches*args.recursion_steps):.4f} "
              f"J2={epoch_j2/(n_batches*args.recursion_steps):.4f} "
              f"J3={epoch_j3/(n_batches*args.recursion_steps):.4f} | "
              f"omegas(Hz)=[{omega_str}]")
        print(f"{'='*80}")

        # ---- Learning Rate Scheduler ----
        scheduler.step(avg_loss)
        current_lr = optimizer.optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.2e}")

        # ---- Save checkpoint ----
        if (epoch + 1) % args.save_interval == 0:
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'criterion_state': criterion.state_dict(),
            }
            path = os.path.join(args.save_dir, f'epoch{epoch+1}.pth.tar')
            torch.save(save_dict, path)
            print(f"Saved checkpoint: {path}")

        # Best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'criterion_state': criterion.state_dict(),
            }
            best_path = os.path.join(args.save_dir, 'best_model.pth.tar')
            torch.save(best_dict, best_path)
            print(f"Best model (loss={best_loss:.6f}) saved.")

    print(f"Training complete. Best loss: {best_loss:.6f}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Neural SVMD — Unsupervised Mode Decomposition Training')

    # Data
    parser.add_argument('--signal_length', type=int, default=16000, help='signal length in samples')
    parser.add_argument('--sample_rate', type=int, default=8000, help='sample rate (Hz)')
    parser.add_argument('--min_modes', type=int, default=3, help='minimum number of modes')
    parser.add_argument('--max_modes', type=int, default=10, help='maximum number of modes')
    parser.add_argument('--noise_std', type=float, default=0.02, help='noise std dev')

    # Model
    parser.add_argument('--n_fft', type=int, default=512, help='STFT window size')
    parser.add_argument('--hop_length', type=int, default=128, help='STFT hop length')
    parser.add_argument('--hidden_dim', type=int, default=64, help='feature dimension')

    # Training
    parser.add_argument('--epochs', type=int, default=200, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--batches_per_epoch', type=int, default=1000, help='batches per epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='L2 regularization')
    parser.add_argument('--max_norm', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='LR warmup steps')
    parser.add_argument('--recursion_steps', type=int, default=20, help='max recursion steps (hard cap)')
    parser.add_argument('--stop_epsilon', type=float, default=0.01,
                        help='adaptive stop threshold: residual energy / original energy')

    # Loss weights (J1/J2/J3 三项量级平衡; 频率轴统一为归一化 [0,0.5])
    # w1 控制带宽紧致性, w2/w3 控制分离力度
    parser.add_argument('--weight_bandwidth', type=float, default=40.0,
                        help='w1: J1 bandwidth compactness weight')
    parser.add_argument('--alpha_bandwidth', type=float, default=50.0,
                        help='beta filter sharpness for J2/J3 (paper alpha, normalized axis)')
    parser.add_argument('--gamma_residual', type=float, default=20.0, help='w2: J2 residual exclusion weight')
    parser.add_argument('--delta_history', type=float, default=20.0, help='w3: J3 history orthogonality weight')

    # System
    parser.add_argument('--use_cuda', type=int, default=1, help='use GPU')
    parser.add_argument('--use_amp', type=int, default=0,
                        help='use AMP mixed precision (default 0=FP32, 更稳防NaN)')
    parser.add_argument('--save_dir', type=str, default='../exp/vmd', help='checkpoint save directory')
    parser.add_argument('--save_interval', type=int, default=10, help='checkpoint interval (epochs)')
    parser.add_argument('--log_interval', type=int, default=50, help='log interval (batches)')
    parser.add_argument('--resume', type=str, default='', help='resume from checkpoint')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("=" * 80)
    print("Neural SVMD — Frequency-Domain Unsupervised Mode Decomposition")
    print("=" * 80)
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("=" * 80)
    train(args)