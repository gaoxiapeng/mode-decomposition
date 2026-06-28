# Neural SVMD — frequency-domain decomposition model
# STFT → frequency U-Net → complex mask → ISTFT

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


# ============================================================================
# U-Net building blocks
# ============================================================================

class FreqEncoderBlock(nn.Module):
    """Conv block along time-frequency axes. Optionally downsamples frequency by 2."""

    def __init__(self, in_ch, out_ch, stride_freq=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, (3, 3),
                               stride=(1, stride_freq),
                               padding=(1, 1), bias=False)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, (3, 3),
                               padding=(1, 1), bias=False)
        self.norm2 = nn.BatchNorm2d(out_ch)

        # 1×1 projection for residual when dimensions change
        self.downsample = None
        if stride_freq > 1 or in_ch != out_ch:
            self.downsample = nn.Conv2d(in_ch, out_ch, 1,
                                        stride=(1, stride_freq), bias=False)

    def forward(self, x):
        residual = self.downsample(x) if self.downsample is not None else x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + residual)


class FreqDecoderBlock(nn.Module):
    """Upsample frequency by 2, concat skip connection, then conv."""

    def __init__(self, in_ch, out_ch, skip_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=(1, 2),
                                     mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, (3, 3),
                               padding=(1, 1), bias=False)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, (3, 3),
                               padding=(1, 1), bias=False)
        self.norm2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, skip):
        x = self.upsample(x)
        # Crop to match skip's frequency size (handles odd dimensions)
        if x.shape[-1] > skip.shape[-1]:
            x = x[:, :, :, :skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out)


class FreqResBlock(nn.Module):
    """Residual block with optional dilation along frequency axis (for bottleneck)."""

    def __init__(self, channels, kernel_size=(3, 3), dilation=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size,
                               padding=(kernel_size[0] // 2,
                                        dilation[1] * (kernel_size[1] // 2)),
                               dilation=dilation, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size,
                               padding=(kernel_size[0] // 2,
                                        dilation[1] * (kernel_size[1] // 2)),
                               dilation=dilation, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + residual)


# ============================================================================
# Main model: Neural SVMD
# ============================================================================

class NeuralSVMD(nn.Module):
    """
    Frequency-domain neural variational mode decomposition with U-Net backbone.

    Architecture:
        STFT → [real, imag, freq_grid] → encoder (freq ↓) → bottleneck
        → decoder (freq ↑ + skip) → single mask M → mode = M·X, residual = (1-M)·X
        → ISTFT → [mode, residual]

    单掩码结构: 网络只预测一个掩码 M ∈ [0,1]. mode 与 residual 互补
    (residual = (1-M)·X), 由于 ISTFT 线性, mode + residual = input 结构性精确成立,
    无需重构损失. 输出仍为 [B, 2, T] (通道0=模态, 通道1=残差) 以兼容下游.

    Args:
        n_fft:         STFT window size (default 512, Δf ≈ 15.6 Hz at 8 kHz)
        hop_length:    STFT hop size (default 128, 75% overlap)
        hidden_dim:    base channel count (default 64)
    """

    def __init__(self, n_fft=512, hop_length=128, hidden_dim=64, sample_rate=8000,
                 max_steps=20):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.max_steps = max_steps
        n_freqs = n_fft // 2 + 1

        # STFT / ISTFT window
        self.register_buffer('window', torch.hann_window(n_fft))

        # Frequency grid: 使用 FFT 归一化频率 [0, 0.5]
        freq_grid = torch.fft.rfftfreq(n_fft, dtype=torch.float32)
        self.register_buffer('freq_grid', freq_grid)  # [n_freqs]

        H = hidden_dim

        # ---- Step embedding (injected at bottleneck) ----
        self.step_embed = nn.Embedding(max_steps, H * 4)

        # ---- Encoder (frequency downsampling) ----
        self.enc0 = FreqEncoderBlock(3,     H,     stride_freq=1)   # → [B, H,   Tf, F]
        self.enc1 = FreqEncoderBlock(H,     H * 2, stride_freq=2)   # → [B, 2H,  Tf, F//2]
        self.enc2 = FreqEncoderBlock(H * 2, H * 4, stride_freq=2)   # → [B, 4H,  Tf, F//4]

        # ---- Bottleneck (dilated ResBlocks at coarsest scale) ----
        self.bottleneck = nn.Sequential(
            FreqResBlock(H * 4, (3, 3), dilation=(1, 1)),
            FreqResBlock(H * 4, (3, 3), dilation=(1, 2)),
            FreqResBlock(H * 4, (3, 3), dilation=(1, 4)),
        )

        # ---- Decoder (frequency upsampling + skip connections) ----
        self.dec2 = FreqDecoderBlock(H * 4, H * 2, H * 2)  # + skip from enc1
        self.dec1 = FreqDecoderBlock(H * 2, H,     H)      # + skip from enc0

        # ---- Output: single mask M ∈ [0,1] (residual uses 1-M) ----
        self.output_conv = nn.Sequential(
            nn.Conv2d(H, H, (3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(H),
            nn.ReLU(),
            nn.Conv2d(H, 1, 1),
            nn.Sigmoid(),
        )

    def _stft(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return torch.stft(x, self.n_fft, self.hop_length,
                          window=self.window, return_complex=True)

    def _istft(self, spec, length):
        return torch.istft(spec, self.n_fft, self.hop_length,
                           window=self.window, length=length)

    def forward(self, x, step):
        """
        Args:
            x:    [B, T] input waveform
            step: int or [B] tensor, current recursion step index (0-indexed)
        Returns:
            dict with keys:
                "mode":         [B, T] extracted mode waveform
                "residual":     [B, T] residual waveform
                "mode_spec":    [B, F, Tf] complex mode spectrum
                "residual_spec":[B, F, Tf] complex residual spectrum
                "mask":         [B, 1, F, Tf] predicted mask M ∈ [0,1]
        """
        B, T = x.shape

        # ---- Step embedding ----
        if isinstance(step, int):
            step = torch.full((B,), step, device=x.device, dtype=torch.long)
        step_emb = self.step_embed(step)                     # [B, 4H]
        step_emb = step_emb[:, :, None, None]                # [B, 4H, 1, 1]

        # ---- STFT ----
        spec = self._stft(x)                            # [B, F, Tf] complex
        F_bins, T_frames = spec.shape[1], spec.shape[2]

        # ---- Prepare input: [real, imag, freq_grid] ----
        spec_2ch = torch.view_as_real(spec)              # [B, F, Tf, 2]
        spec_2ch = spec_2ch.permute(0, 3, 2, 1)          # [B, 2, Tf, F]

        freq_ch = ((self.freq_grid[:F_bins] / 0.5)
                   .view(1, 1, 1, F_bins)
                   .expand(B, 1, T_frames, F_bins))      # [B, 1, Tf, F], normalized [0,1]

        feat = torch.cat([spec_2ch, freq_ch], dim=1)     # [B, 3, Tf, F]

        # ---- U-Net ----
        e0 = self.enc0(feat)        # [B, H,  Tf, F]
        e1 = self.enc1(e0)          # [B, 2H, Tf, F//2]
        e2 = self.enc2(e1)          # [B, 4H, Tf, F//4]

        b = self.bottleneck(e2)     # [B, 4H, Tf, F//4]

        # Inject step embedding at bottleneck
        b = b + step_emb            # [B, 4H, Tf, F//4]

        d2 = self.dec2(b, e1)       # [B, 2H, Tf, F//2]
        d1 = self.dec1(d2, e0)      # [B, H,  Tf, F]

        # ---- Output: single mask M ∈ [0,1] ----
        mask = self.output_conv(d1)                      # [B, 1, Tf, F]
        mask_t = mask.transpose(2, 3).contiguous()       # [B, 1, F, Tf]

        # ---- Apply complementary masks to complex spectrum ----
        # mode = M·X, residual = (1-M)·X  →  mode + residual = X (structural)
        mode_spec = spec * mask_t[:, 0, :, :]             # [B, F, Tf] complex
        resid_spec = spec * (1.0 - mask_t[:, 0, :, :])    # [B, F, Tf] complex

        # ---- ISTFT ----
        mode_wav = self._istft(mode_spec, T)              # [B, T]
        resid_wav = self._istft(resid_spec, T)            # [B, T]

        return {
            "mode": mode_wav,              # [B, T]
            "residual": resid_wav,         # [B, T]
            "mode_spec": mode_spec,        # [B, F, Tf] complex
            "residual_spec": resid_spec,   # [B, F, Tf] complex
            "signal_spec": spec,           # [B, F, Tf] complex (input STFT, 避免外部重复计算)
            "mask": mask_t,                # [B, 1, F, Tf]
        }
