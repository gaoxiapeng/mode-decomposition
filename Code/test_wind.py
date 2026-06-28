# Neural SVMD — 真实风电信号模态分解测试
#
# 读取 exp/wind 下 excel 的第 4 列(信号)，递归分解并可视化：
#   第 1 行: 原始风电信号 (左=时域, 右=频域)
#   后续行: 分解出的各模态 (左=时域, 右=频域)
#   末 行: 最终残差 (黑色)
#
# 用法:
#   cd Code && python test_wind.py \
#       --excel ../exp/wind/group_4_r10r30.xlsx \
#       --checkpoint ../exp/vmd_v4/best_model.pth.tar \
#       --sample_rate 8000 \
#       --out ../exp/wind/wind_decomp.png

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


# ============================================================================
# 读取 excel 第 col_idx 列 (0-based)
# ============================================================================

def read_excel_column(path, col_idx=3):
    """读取 xlsx 指定列(默认第4列, 0-based=3)。优先 openpyxl, 回退到无依赖 zip/xml 解析。"""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows = ws.iter_rows(values_only=True)
        header = next(rows)
        col_name = header[col_idx] if header and col_idx < len(header) else f'col{col_idx}'
        vals = []
        for r in rows:
            if col_idx < len(r) and isinstance(r[col_idx], (int, float)):
                vals.append(float(r[col_idx]))
        return np.asarray(vals, dtype=np.float32), str(col_name)
    except ModuleNotFoundError:
        pass

    # ---- 无依赖回退: xlsx = zip(xml) ----
    import zipfile, re
    from xml.etree import ElementTree as ET
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        # 共享字符串(列名可能在这里)
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            st = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in st.findall('m:si', ns):
                shared.append(''.join(t.text or '' for t in si.iter('{%s}t' % ns['m'])))
        sheet = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))

    def col_letters(ref):
        return re.match(r'[A-Z]+', ref).group()
    def col_num(letters):
        n = 0
        for ch in letters:
            n = n * 26 + (ord(ch) - ord('A') + 1)
        return n - 1  # 0-based

    data, col_name = [], f'col{col_idx}'
    for ri, row in enumerate(sheet.iter('{%s}row' % ns['m'])):
        for c in row.findall('m:c', ns):
            ref = c.get('r')
            if col_num(col_letters(ref)) != col_idx:
                continue
            v = c.find('m:v', ns)
            if v is None:
                continue
            if c.get('t') == 's':
                txt = shared[int(v.text)]
                if ri == 0:
                    col_name = txt
            else:
                if ri == 0:
                    col_name = str(v.text)
                else:
                    data.append(float(v.text))
    return np.asarray(data, dtype=np.float32), col_name


# ============================================================================
# 递归分解 (复用 test_vmd 的三重停止判据)
# ============================================================================

@torch.no_grad()
def decompose(model, signal, device, sr, max_steps=20, epsilon=0.005,
              plateau_ratio=0.02, dup_freq_hz=None):
    """残差递归分解。三重停止: 能量 / 平台 / 重复频率。"""
    if dup_freq_hz is None:
        dup_freq_hz = sr / 512.0 * 3  # ≈3 个频率分辨率单元

    x = torch.from_numpy(signal).float().unsqueeze(0).to(device)
    original_energy = torch.mean(x ** 2).item()

    modes, center_freqs = [], []
    cur = x
    prev_res_energy = original_energy

    for step in range(max_steps):
        out = model(cur, step)
        u = out['mode'][0, :].cpu().numpy()
        f_r = out['residual']

        cf = compute_center_freq(out['mode'], sample_rate=sr).item()
        res_energy = compute_residual_energy(f_r).item()
        rel = res_energy / (original_energy + 1e-8)

        if center_freqs and min(abs(cf - c) for c in center_freqs) < dup_freq_hz:
            break

        center_freqs.append(cf)
        modes.append(u)
        print(f'  step{step+1}: CF={cf:7.1f}Hz  rel_res={rel:.4f}')

        if rel < epsilon:
            print('  -> stop: residual below epsilon')
            break
        drop = (prev_res_energy - res_energy) / (prev_res_energy + 1e-8)
        if step > 0 and drop < plateau_ratio:
            print('  -> stop: residual plateaued')
            break
        prev_res_energy = res_energy
        cur = f_r

    return modes, center_freqs, f_r.squeeze().cpu().numpy()


# ============================================================================
# 绘图
# ============================================================================

def plot_all(signal, modes, center_freqs, residual, sr, out_path, sig_name):
    T = len(signal)
    t_axis = np.arange(T) / sr
    freqs = np.fft.rfftfreq(T, 1.0 / sr)
    win = np.hanning(T)

    n_rows = 1 + len(modes) + 1  # 原始 + 各模态 + 残差
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 2.2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    def row(idx, data, title, color):
        at, af = axes[idx, 0], axes[idx, 1]
        at.plot(t_axis, data, color=color, linewidth=0.5)
        at.set_ylabel('Amplitude'); at.set_title(title, fontsize=9)
        at.grid(True, alpha=0.3); at.set_xlim(0, t_axis[-1])
        sp = np.abs(np.fft.rfft(data * win))
        af.plot(freqs, sp, color=color, linewidth=0.5)
        af.set_ylabel('Magnitude'); af.set_title(title, fontsize=9)
        af.grid(True, alpha=0.3); af.set_xlim(0, sr / 2)

    colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#e377c2', '#7f7f7f',
              '#d62728', '#8c564b', '#bcbd22', '#17becf', '#1f77b4',
              '#1a9850', '#fee090']

    # 第 1 行: 原始信号
    row(0, signal, f'Original wind signal — {sig_name}', '#1f77b4')
    # 各模态
    for i, (m, cf) in enumerate(zip(modes, center_freqs)):
        row(1 + i, m, f'Mode {i+1} — CF = {cf:.1f} Hz', colors[i % len(colors)])
    # 末行: 残差
    res_e = float(np.mean(residual ** 2))
    row(n_rows - 1, residual, f'Residual — energy = {res_e:.2e}', 'black')

    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Frequency (Hz)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description='Neural SVMD — 真实风电信号分解')
    ap.add_argument('--excel', default='../exp/wind/group_4_r10r30.xlsx')
    ap.add_argument('--col', type=int, default=3, help='信号列 (0-based, 第4列=3)')
    ap.add_argument('--checkpoint', default='../exp/vmd_v4/best_model.pth.tar')
    ap.add_argument('--sample_rate', type=int, default=8000, help='风电信号采样率(Hz)')
    ap.add_argument('--max_steps', type=int, default=20)
    ap.add_argument('--epsilon', type=float, default=0.005)
    ap.add_argument('--normalize', type=int, default=1, help='是否对信号做标准化(零均值/单位方差)')
    ap.add_argument('--out', default='../exp/wind/wind_decomp.png')
    ap.add_argument('--use_cuda', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    # 读信号
    signal, col_name = read_excel_column(args.excel, args.col)
    print(f'信号列: {col_name!r}  长度: {len(signal)}  '
          f'范围: [{signal.min():.3f}, {signal.max():.3f}]')

    # 去均值 (+可选标准化)。模型在零均值合成信号上训练, 真实信号常有直流分量。
    signal = signal - signal.mean()
    if args.normalize:
        std = signal.std()
        if std > 1e-8:
            signal = signal / std

    # 模型
    model = NeuralSVMD(n_fft=512, hop_length=128, hidden_dim=64,
                       sample_rate=args.sample_rate)
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print(f'模型: {args.checkpoint} (epoch {ckpt.get("epoch", "?")})')

    # 分解
    modes, cfs, residual = decompose(
        model, signal, device, args.sample_rate,
        max_steps=args.max_steps, epsilon=args.epsilon)
    print(f'提取模态数: {len(modes)}  CFs: {[f"{c:.1f}" for c in cfs]}')

    recon = np.sum(modes, axis=0) + residual
    print(f'重构 MSE: {np.mean((signal - recon) ** 2):.6e}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plot_all(signal, modes, cfs, residual, args.sample_rate, args.out, col_name)


if __name__ == '__main__':
    main()
