# 风电功率信号 → 训练切片
#
# 读取 exp/wind/wind100MW_15min.xlsx 的【最后一列】(功率), 滑动窗口切片,
# 每段去均值+标准化, 存成 exp/wind/wind_train.npy  形状 [N, win]
#
# 采样按 1min 处理 (文件名的 15min 忽略)。
#
# 用法:
#   cd Code && python prepare_wind_data.py
#   cd Code && python prepare_wind_data.py --win 1024 --hop 128

import os
import sys
import argparse
import numpy as np


def read_excel_last_column(path):
    """无依赖读取 xlsx 最后一列的数值(功率)。返回 (values[np.float32], col_name)。"""
    import zipfile, re
    from xml.etree import ElementTree as ET
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

    with zipfile.ZipFile(path) as z:
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

    rows = list(sheet.iter('{%s}row' % ns['m']))
    # 末列索引: 扫表头行
    max_col = 0
    for c in rows[0].findall('m:c', ns):
        max_col = max(max_col, col_num(col_letters(c.get('r'))))

    col_name = f'col{max_col}'
    data = []
    for ri, row in enumerate(rows):
        for c in row.findall('m:c', ns):
            if col_num(col_letters(c.get('r'))) != max_col:
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
                    col_name = str(v.text)  # 表头是数字的情况
                else:
                    data.append(float(v.text))
    return np.asarray(data, dtype=np.float32), col_name


def slice_windows(sig, win, hop, normalize, min_std):
    """对一段连续信号滑窗切片。返回 (arr[N, win], n_skip)。段数为 0 时 arr 为空数组。"""
    segs = []
    n_skip = 0
    for start in range(0, len(sig) - win + 1, hop):
        seg = sig[start:start + win].copy()
        if normalize:
            seg = seg - seg.mean()
            s = seg.std()
            if s < min_std:      # 平直段无信息, 跳过
                n_skip += 1
                continue
            seg = seg / s
        segs.append(seg)
    arr = (np.stack(segs).astype(np.float32) if segs
           else np.empty((0, win), dtype=np.float32))
    return arr, n_skip


def main():
    ap = argparse.ArgumentParser(description='风电功率信号 → 滑窗训练/测试切片 (按时间划分)')
    ap.add_argument('--excel', default='../exp/wind/wind100MW_15min.xlsx')
    ap.add_argument('--win', type=int, default=1024, help='窗口长度(点)')
    ap.add_argument('--hop', type=int, default=128, help='滑动步长(点)')
    ap.add_argument('--out_train', default='../exp/wind/wind_train.npy')
    ap.add_argument('--out_test', default='../exp/wind/wind_test.npy')
    ap.add_argument('--test_ratio', type=float, default=0.2,
                    help='末尾用作测试集的时间比例 (0=不划分, 全部做训练)')
    ap.add_argument('--gap', type=int, default=-1,
                    help='train/test 之间丢弃的间隔点数, 防重叠窗口跨界泄漏; '
                         '-1=自动取一个 win 长')
    ap.add_argument('--normalize', type=int, default=1,
                    help='每段去均值并除以标准差 (1=是)')
    ap.add_argument('--min_std', type=float, default=1e-6,
                    help='标准差低于此值的段视为平直/无效, 丢弃')
    args = ap.parse_args()

    sig, col_name = read_excel_last_column(args.excel)
    print(f'功率列: {col_name!r}  总点数: {len(sig)}  '
          f'范围: [{np.nanmin(sig):.3f}, {np.nanmax(sig):.3f}]')

    # 去除 NaN(用前向填充, 简单稳健)
    if np.isnan(sig).any():
        n_nan = int(np.isnan(sig).sum())
        idx = np.where(~np.isnan(sig))[0]
        sig = np.interp(np.arange(len(sig)), idx, sig[idx]).astype(np.float32)
        print(f'  插值填补了 {n_nan} 个 NaN')

    # ---- 按时间划分 train / test (无重叠, 中间留 gap 防泄漏) ----
    # 时序数据不能随机划分: 相邻滑窗高度重叠, 随机切会让测试段信息泄漏进训练。
    # 故按时间顺序: 前 (1-test_ratio) 做训练, 末尾 test_ratio 做测试, 中间丢一个 win。
    gap = args.win if args.gap < 0 else args.gap
    n = len(sig)
    if args.test_ratio <= 0:
        train_sig, test_sig = sig, np.empty(0, dtype=sig.dtype)
    else:
        n_test = int(round(n * args.test_ratio))
        split = n - n_test
        train_sig = sig[:max(0, split - gap)]   # 训练段末尾丢 gap 点
        test_sig = sig[split:]
        print(f'时间划分: 训练 [0:{max(0, split-gap)}]  '
              f'间隔丢弃 {gap} 点  测试 [{split}:{n}]')

    train_arr, tr_skip = slice_windows(train_sig, args.win, args.hop,
                                       args.normalize, args.min_std)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_train)), exist_ok=True)
    np.save(args.out_train, train_arr)
    overlap = 100 * (1 - args.hop / args.win)
    print(f'[train] {train_arr.shape[0]} 段 × {args.win} 点 (步长 {args.hop}, '
          f'重叠 {overlap:.0f}%)  跳过平直段 {tr_skip}  →  {args.out_train}')

    if args.test_ratio > 0:
        test_arr, te_skip = slice_windows(test_sig, args.win, args.hop,
                                          args.normalize, args.min_std)
        np.save(args.out_test, test_arr)
        print(f'[test]  {test_arr.shape[0]} 段 × {args.win} 点  '
              f'跳过平直段 {te_skip}  →  {args.out_test}')


if __name__ == '__main__':
    main()
