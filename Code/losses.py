# Neural SVMD v2 — 损失函数重构
# 严格遵循 SVMD 原论文目标函数，同时适用于神经网络端到端训练
#
# 损失全部工作于频域（直接使用网络输出的频谱，不再重复 FFT）
# J1: 频谱紧致性 (论文 Eq.3, Parseval 形式)
# J2: 残差排除 (论文 Eq.5, β 滤波器)
# J3: 历史正交 (论文 Eq.7, β 滤波器)
#
# 中心频率 ω 维护在 NeuralSVMDCriterion 的 buffer 中，
# 按论文 ADMM 交替更新思想，使用 EMA 在 epoch 结束时更新。

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


# ============================================================================
# 辅助函数（推理脚本使用）
# ============================================================================

def _safe_power(fft_result):
    """Compute power spectrum safely."""
    return torch.abs(fft_result) ** 2 + EPS


def _freq_grid(n_fft, device):
    """Normalized frequency grid [0, 0.5] for rFFT output — 使用 FFT 实际频率."""
    return torch.fft.rfftfreq(n_fft, device=device)[:n_fft]


def _hann_window(signal_length, device):
    """Hann window for reducing spectral leakage."""
    return torch.hann_window(signal_length, device=device)


# ============================================================================
# 推理辅助函数（仅用于显示和停止判断，不参与 Loss）
# ============================================================================

def compute_center_freq(u, sample_rate=1.0):
    """
    计算模态的谱质心 (中心频率) — 仅用于推理显示.

    公式: ω_L = Σ ω·|û(ω)|² / Σ |û(ω)|²

    Args:
        u: [B, T] 模态信号
        sample_rate: 采样率 (默认1.0 = 归一化频率 [0, 0.5])
    Returns:
        center_freq: [B] 每个 batch 样本的中心频率
    """
    B, T = u.shape
    n_fft = T // 2 + 1

    window = _hann_window(T, u.device)
    u_fft = torch.fft.rfft(u * window, dim=-1)
    power = _safe_power(u_fft)  # [B, n_fft]

    freqs = torch.fft.rfftfreq(T, device=u.device)[:n_fft]  # [n_fft], normalized [0, 0.5]

    total_power = power.sum(dim=-1)  # [B]
    spectral_sum = (freqs.unsqueeze(0) * power).sum(dim=-1)  # [B]

    center_freq = spectral_sum / (total_power + EPS)
    return center_freq * sample_rate


def compute_residual_energy(residual):
    """
    计算残差能量，用于自适应终止判断.

    Args:
        residual: [B, T] 残差信号
    Returns:
        energy: [B] 每个 batch 样本的残差能量
    """
    return torch.mean(residual ** 2, dim=-1)


# ============================================================================
# NeuralSVMDCriterion — 神经网络 SVMD 损失
# ============================================================================

class NeuralSVMDCriterion(nn.Module):
    """
    神经网络 SVMD 损失准则.

    维护中心频率 ω 为 buffer（非可学习参数），按 ADMM 交替更新思想
    在 epoch 结束时用 EMA 更新. Loss 全部工作于频域.

    Args:
        alpha:  β 滤波器锐度 (论文 α, 越大陷波越窄)
        eps:    数值稳定项
        momentum: ω 的 EMA 动量 (0.95 = 保留 95% 历史)
        w1:     J1 带宽紧致性权重
        w2:     J2 残差排除权重
        w3:     J3 历史正交权重
    """

    def __init__(self, alpha=50.0, eps=1e-8, momentum=0.95,
                 w1=20.0, w2=1.0, w3=2.0, sample_rate=8000, max_steps=20):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.sample_rate = sample_rate
        self.max_steps = max_steps

        # 中心频率 ω: 每个递归 step 独立维护 (非 Parameter), 向量 [max_steps]
        # 符合 SVMD「每个模态拥有独立中心频率」的设计
        self.register_buffer('omegas', torch.zeros(max_steps))
        # 各 step 是否已初始化
        self.register_buffer('_omega_init', torch.zeros(max_steps, dtype=torch.long))
        # 各 step 的 epoch 内累积 (按 step 分桶)
        self.register_buffer('_centroid_sum', torch.zeros(max_steps))
        self.register_buffer('_centroid_n', torch.zeros(max_steps, dtype=torch.long))

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _freq_grid_for(self, n_freq, device):
        """FFT 归一化频率轴 [0, 0.5]. n_freq = n_fft//2 + 1."""
        n_fft = (n_freq - 1) * 2
        return torch.fft.rfftfreq(n_fft, device=device)  # 归一化 [0, 0.5]

    def _compute_centroid(self, mode_spec):
        """
        从复频谱计算谱质心.

        Args:
            mode_spec: [B, F, Tf] complex tensor
        Returns:
            centroid: [B] per-sample centroid
        """
        power = torch.abs(mode_spec) ** 2 + self.eps           # [B, F, Tf]
        power_f = power.sum(dim=-1)                             # [B, F] — 时间求和(论文二维积分)
        n_freq = power_f.shape[-1]
        freq_grid = self._freq_grid_for(n_freq, mode_spec.device)  # [F] Hz

        total = power_f.sum(dim=-1)                             # [B]
        weighted = (freq_grid.unsqueeze(0) * power_f).sum(dim=-1)  # [B]
        return weighted / (total + self.eps)

    def _clamp_step(self, step):
        """把 step 限制在 [0, max_steps-1], 防止递归超过 max_steps 时索引越界."""
        if step < 0:
            return 0
        if step >= self.max_steps:
            return self.max_steps - 1
        return step

    def accumulate_centroid(self, centroid, step):
        """在 forward 中累积 batch 平均质心(按 step 分桶), 供 epoch 结束 update_omega 使用."""
        step = self._clamp_step(step)
        self._centroid_sum[step] += centroid.detach().mean()
        self._centroid_n[step] += 1

    # ------------------------------------------------------------------
    # ω 更新 (每次递归 step 后调用)
    # ------------------------------------------------------------------

    def update_omega(self, step):
        """
        EMA 更新第 step 步的中心频率 omegas[step] (每次递归 step 后调用).

        omegas[step] = momentum * omegas[step] + (1-momentum) * centroid_avg
        """
        step = self._clamp_step(step)
        if self._centroid_n[step] > 0:
            avg_centroid = self._centroid_sum[step] / self._centroid_n[step]
            self.omegas[step] = (
                self.momentum * self.omegas[step] +
                (1.0 - self.momentum) * avg_centroid
            )
            # 重置该 step 的累积器
            self._centroid_sum[step] = 0.0
            self._centroid_n[step] = 0

    # ------------------------------------------------------------------
    # J1 — 频谱紧致性 (论文 Eq.3, Parseval 形式)
    # ------------------------------------------------------------------

    def bandwidth_loss(self, mode_spec, omega):
        """
        J1 — 频谱紧致性损失.

        原论文: J1 = ||∂_t[u_A(t)·exp(-jω_L t)]||²
        Parseval → J1 = 4α ∫(ω-ω_L)²|û(ω)|² dω

        本实现: 功率归一化的频谱二阶矩, 度量频谱"集中度".
        ω_L 取「当前模态自身的谱质心」(论文每次迭代重算), 逐样本 [B].

        Args:
            mode_spec: [B, F, Tf] complex
            omega:     [B] 每个样本的中心频率 (detached)
        Returns:
            J1: scalar loss
        """
        power = torch.abs(mode_spec) ** 2 + self.eps          # [B, F, Tf]
        n_freq = mode_spec.shape[1]
        freq_grid = self._freq_grid_for(n_freq, mode_spec.device)  # [F]

        # 逐样本: (freq_grid[F] - omega[B,1])² → [B, F]
        diff_sq = (freq_grid.unsqueeze(0) - omega.unsqueeze(-1)) ** 2   # [B, F]
        loss = (diff_sq.unsqueeze(-1) * power).sum()
        loss = loss / (power.sum() + self.eps)
        return self.alpha * loss  # 论文 Parseval 后有 4α 系数

    # ------------------------------------------------------------------
    # J2 — 残差排除 (论文 Eq.5, β 滤波器)
    # ------------------------------------------------------------------

    def _beta_filter(self, freq_grid, omega):
        """
        论文 β 滤波器: β̂(ω) = 1 / (α·(ω-ω_L)²), 逐样本归一化到 [0,1].

        归一化避免 ω→ω_L 时的数值爆炸. 用于 J2(残差排除) 与 J3(历史正交).

        Args:
            freq_grid: [F] 频率轴
            omega:     [B] 每个样本的中心频率
        Returns:
            beta: [B, F] ∈ [0,1]
        """
        distance = (freq_grid.unsqueeze(0) - omega.unsqueeze(-1)).pow(2)  # [B, F]
        beta = 1.0 / (self.alpha * (distance + self.eps))                 # [B, F]
        beta = beta / (beta.amax(dim=-1, keepdim=True) + self.eps)        # 逐样本归一化
        return beta

    def residual_loss(self, residual_spec, omega):
        """
        J2 — 残差频谱排除损失.

        论文定义:
            β̂(ω) = 1 / (α·(ω-ω_L)²)
            J2 = ||β * f_r||² = Σ |β̂(ω)|² · |f̂_r(ω)|²

        β 归一化到 [0,1]，避免 r→0 时的数值爆炸.

        Args:
            residual_spec: [B, F, Tf] complex
            omega:         [B] 当前模态(每样本)的中心频率
        Returns:
            J2: scalar loss
        """
        n_freq = residual_spec.shape[1]
        freq_grid = self._freq_grid_for(n_freq, residual_spec.device)  # [F]

        beta = self._beta_filter(freq_grid, omega)               # [B, F]
        power = torch.abs(residual_spec) ** 2                    # [B, F, Tf]
        return (beta.unsqueeze(-1) ** 2 * power).mean()

    # ------------------------------------------------------------------
    # J3 — 历史正交 (论文 Eq.7, β 滤波器)
    # ------------------------------------------------------------------

    def history_loss(self, mode_spec, history):
        """
        J3 — 历史模态正交性损失 (论文 Eq.7, β 滤波器形式).

        对 history 中每个已提取模态, 用其中心频率 omega_i 重建 β 滤波器 β_i,
        再惩罚新模态在 β_i 通带 (即 omega_i 附近) 内的能量:

            J3 = Σ_i mean(|β_i(ω)|² · |û_new(ω)|²)

        即新模态不应落在任一历史模态的中心频率附近 (论文 Eq.7 的 β 排除).

        Args:
            mode_spec: [B, F, Tf] complex, 当前(新)模态频谱
            history:   list of {"mode_spec": [B,F,Tf] complex, "omega": scalar}
        Returns:
            J3: scalar loss (history 为空时返回 0)
        """
        if not history:
            return torch.tensor(0.0, device=mode_spec.device)

        n_freq = mode_spec.shape[1]
        freq_grid = self._freq_grid_for(n_freq, mode_spec.device)  # [F]
        curr_power = torch.abs(mode_spec) ** 2                     # [B, F, Tf]

        loss = 0.0
        for item in history:
            omega_i = item["omega"]                                # [B]
            beta_i = self._beta_filter(freq_grid, omega_i)         # [B, F] ∈[0,1]
            loss += (beta_i.unsqueeze(-1) ** 2 * curr_power).mean()

        return loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, mode_spec, residual_spec, signal_spec, history, step):
        """
        计算复合 SVMD 损失.

        Args:
            mode_spec:     [B, F, Tf] complex, 当前模态频谱
            residual_spec: [B, F, Tf] complex, 当前残差频谱
            signal_spec:   [B, F, Tf] complex, 当前输入频谱 (保留兼容, 当前未使用)
            history:       list of {"omega": scalar, "mode_spec": [B,F,Tf] complex}
            step:          int, 当前递归步索引 (用于选择 omegas[step])

        Returns:
            dict: {"loss", "j1", "j2", "j3", "centroid"}
        """
        s = self._clamp_step(step)

        # 当前 batch 在该 step 的逐样本质心 (论文: ω_L = 当前模态自身谱质心)
        centroid = self._compute_centroid(mode_spec)              # [B]
        self.accumulate_centroid(centroid, step)

        # omegas[step] 仍按 EMA 更新, 但仅用于日志监控, 不再驱动损失
        if not self._omega_init[s]:
            self.omegas[s] = centroid.detach().mean()
            self._omega_init[s] = 1

        # J1/J2 用「当前模态实时质心」(逐样本, detached) 作为 ω_L —
        # 摆脱全局 EMA 低频锁定, 让每个信号的高频模态也能被对应频率参考
        omega = centroid.detach()                                 # [B]

        j1 = self.bandwidth_loss(mode_spec, omega)
        j2 = self.residual_loss(residual_spec, omega)
        j3 = self.history_loss(mode_spec, history)

        loss = self.w1 * j1 + self.w2 * j2 + self.w3 * j3

        return {
            "loss": loss,
            "j1": j1.detach(),
            "j2": j2.detach(),
            "j3": j3.detach(),
            "centroid": centroid.detach(),
        }
