# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Neural SVMD — unsupervised neural variational mode decomposition. A frequency-domain
U-Net (`NeuralSVMD`) takes a 1-D waveform plus a recursion **step index**, predicts a
**single** real mask `M ∈ [0,1]` over its STFT, and returns the extracted mode (`M·X`) and
residual (`(1−M)·X`). Because ISTFT is linear, `mode + residual = input` holds
**structurally** — no reconstruction loss is needed. Decomposition is **recursive**: the
residual from one step becomes the input for the next, extracting one mode per step until
residual energy drops below a threshold. Training is fully unsupervised via a
frequency-domain SVMD-derived composite loss (paper Eq. 9: `min{α·J1 + J2 + J3}`) — no
ground truth is required.

## Commands

```bash
# Train (defaults: 200 epochs, save to ../exp/vmd)
cd Code && python train_vmd.py

# Test on 10 fixed synthetic signals (3–10 modes), save plots to ../exp/vmd/test_output
cd Code && python test_vmd.py --checkpoint ../exp/vmd/best_model.pth.tar

# Demo: recursive inference + plots on an 8-mode signal
cd Code && python demo_vmd.py --model_path ../exp/vmd/best_model.pth.tar

# Quick end-to-end smoke test: trains a tiny model on CPU (~5-10 min) then decomposes
cd Code && python demo_lightweight.py            # add --skip_train to only run inference

# Check a checkpoint for NaN/Inf weights
cd Code && python check_model_weights.py <checkpoint.pth.tar>
```

There is no linter or test framework. Verification is visual (the plotting scripts) plus
the reconstruction MSE printed by `test_vmd.py`.

## Architecture

### Core model (`Code/models.py`) — STFT U-Net

`NeuralSVMD(n_fft=512, hop_length=128, hidden_dim=64, sample_rate=8000, max_steps=20)`.
`forward(x, step)` takes a waveform `[B, T]` and a recursion `step` (int or `[B]` long
tensor) and returns a **dict**, not a stacked tensor:

1. **Step embedding**: `nn.Embedding(max_steps, 4H)` maps `step` → a vector added to the
   bottleneck. This lets the network condition its mask on recursion depth.
2. **STFT** of input `[B, T]` → complex spectrum `[B, F, Tf]`.
3. Build a 3-channel feature map `[real, imag, freq_grid]` shaped `[B, 3, Tf, F]`. The
   `freq_grid` channel is the rFFT frequency axis rescaled to `[0, 1]` (i.e. `[0,0.5]/0.5`),
   a positional encoding along frequency.
4. **Encoder** (`FreqEncoderBlock` ×3): residual conv blocks that downsample the
   *frequency* axis by 2 each (time axis untouched). Channels `H → 2H → 4H`.
5. **Bottleneck**: three `FreqResBlock`s with dilation `(1,1)/(1,2)/(1,4)` along frequency;
   the step embedding is added here.
6. **Decoder** (`FreqDecoderBlock` ×2): bilinear upsample on frequency + U-Net skip
   connections from the matching encoder stage, then conv.
7. **Output**: a conv head with **`Sigmoid`** produces a **single** mask `M ∈ [0,1]`
   shaped `[B, 1, F, Tf]`.
8. `mode_spec = M·X`, `resid_spec = (1−M)·X`; **ISTFT** of each gives `mode_wav` and
   `resid_wav`.

The returned dict has keys: `mode` `[B,T]`, `residual` `[B,T]`, `mode_spec` `[B,F,Tf]`
complex, `residual_spec` `[B,F,Tf]` complex, `signal_spec` `[B,F,Tf]` (input STFT, exposed
so callers don't recompute it), and `mask` `[B,1,F,Tf]`.

The mode and residual are **complementary by construction** (`M` and `1−M`), so
`mode + residual = input` holds exactly. Reconstruction is therefore **structural, not
learned** — there is no reconstruction loss term.

### Losses (`Code/losses.py`)

The training loss is a **stateful `nn.Module`**, `NeuralSVMDCriterion`, not a set of free
functions. It maintains the center frequency `ω` as a non-learnable **buffer** updated by
EMA — mimicking the paper's ADMM alternating updates. All loss terms operate directly on the
complex spectra the model already returns (`mode_spec`, `residual_spec`), so no extra FFT is
done inside the loss. The composite single-step loss:

```
L_step = w1·J1 + w2·J2 + w3·J3
```

`w1` (default **20**) balances J1 against J2/J3. Unlike the paper's ADMM solver (which
structurally removes extracted modes from the residual), the neural recursive framework
relies on J2/J3 for diversity. If `w1` is too large it suppresses J2/J3 and the model
collapses to a fixed bandpass filter extracting the same center frequency every step.

| Loss | Method | Paper | Purpose |
|------|--------|-------|---------|
| J1 | `bandwidth_loss(mode_spec, omega)` | Eq. 3 | Bandwidth compactness — power-normalized spectral 2nd moment `Σ(ω−ω_L)²·power / Σpower`, scaled by `alpha`. Concentrates mode energy near `ω`. |
| J2 | `residual_loss(residual_spec, omega)` | Eq. 5 | Residual separation — β-filter `β̂(ω)=1/(α·(ω−ω_L)²)` normalized to `[0,1]` by its max, then `mean(β² · |residual|²)`. Penalizes residual energy near `ω`. |
| J3 | `history_loss(mode_spec, history)` | Eq. 7 | History separation — for each past mode rebuild its β filter from that mode's own `ω_i`, then `Σ_i mean(β_i² · |û_new|²)`. Penalizes new-mode energy near any previously-extracted center freq. 0 when history empty. |

### How frequency / spectrum / centroid are computed (read this before editing the loss)

- **Spectrum**: the model returns STFT complex spectra `[B,F,Tf]` (Hann-windowed). The loss
  consumes these directly. Inference helpers instead do a single Hann-windowed `rfft` over the
  whole waveform `[B,F]`. **Power spectrum is always `|spec|²`**.
- **Frequency axis is NORMALIZED, not Hz**: every grid is `torch.fft.rfftfreq(n_fft)` ∈ `[0,0.5]`
  (cycles/sample). `sample_rate` is **not** used inside the model or the loss — it only multiplies
  the axis for Hz-valued *display* in `compute_center_freq` / plotting. So the model operates on
  normalized frequency; changing `--sample_rate` changes labels, not behavior.
- **Center frequency `ω` is a spectral centroid (power-weighted mean frequency), NOT a peak/argmax
  or any count-based statistic**: `ω = Σ_f(freq[f]·power[f]) / Σ_f power[f]`. `_compute_centroid`
  (training) first sums power over time → `[B,F]`, then takes the centroid → `[B]`;
  `compute_center_freq` (display) is the same formula × `sample_rate`.
  - Caveat: a centroid lands at the energy *barycenter*. For an impure multi-peak mode it falls in
    the empty gap between peaks (the old test_10 bug: a mode with peaks at 1000/1800/2400/3200 had
    `ω≈1899`, where there is no energy). Don't judge mode quality by CF alone — check the spectrum's
    peak count.

ω is no longer a buffer-driven reference (that caused low-freq lock-in — see `改动清单.md` round 2):
- `forward(mode_spec, residual_spec, signal_spec, history, step)` computes the **per-sample
  centroid** `[B]` of the current mode and uses `omega = centroid.detach()` to drive J1/J2 — so the
  reference frequency tracks each signal's current mode, not a global average.
- The `omegas[max_steps]` buffer + `update_omega(step)` EMA still exist but are **monitoring-only**
  (logged per epoch), they do not drive the loss.
- `history` stores each step's per-sample centroid as `omega` `[B]`, which J3 uses to rebuild β_i.
- `forward` returns `{"loss", "j1", "j2", "j3", "centroid"}` — J-terms are `.detach()`ed (display),
  `centroid` `[B]` is detached and appended to `history` by the training loop.

Key conventions:
- **All loss terms share one normalized rFFT axis** (`_freq_grid_for` → `torch.fft.rfftfreq(n_fft)`,
  ∈`[0,0.5]`), derived from the spectrum's bin count, so `alpha` has one meaning across J1/J2/J3.
- The per-batch centroid is always `.detach()`ed before it touches J1/J2/history — gradients flow
  through the spectrum, not the centroid.
- J1/J2/J3 are **power-normalized** (divided by total power / mean), making them shape metrics
  decoupled from absolute mode energy. (Paper's J1 is unnormalized; normalization is a
  deliberate stabilization choice here.)
- J2's β-filter is **max-normalized to `[0,1]`** (rather than clamped) to avoid the `r→0`
  numerical explosion of the raw `1/(α·r²)` form.
- **No reconstruction loss** and **no energy-floor term**: reconstruction is structural (see
  Architecture); the loss has exactly three terms J1/J2/J3.

Inference-only helpers (also in `losses.py`, NOT part of the criterion):
- `compute_center_freq(u, sample_rate=1.0)` — power-weighted spectral centroid of a waveform,
  used by inference scripts for Hz-valued display (pass the real `sample_rate`).
- `compute_residual_energy(residual)` — mean-square energy, the adaptive stopping signal used
  by all inference scripts and the training loop.

### Training (`Code/train_vmd.py`)

- **Data**: `generate_synthetic_signal` (defined inline in this file) makes on-the-fly
  multi-component AM-FM signals, 3–10 modes, enforced min frequency spacing, light noise.
  Batches are generated directly each step (no DataLoader).
- **Recursive loop**: instantiates a `NeuralSVMDCriterion`; each step calls
  `model(current_input, step)` → `criterion(mode_spec, residual_spec, signal_spec, history)`,
  then appends `{"omega", "mode_spec"}` to `history` (for J3), calls `criterion.update_omega()`,
  and feeds the residual to the next step. Gradients flow through the whole recursive chain.
  The step loss is **averaged over `actual_steps`**. **Adaptive stop**: breaks early once every
  sample's residual energy < `stop_epsilon · original_energy`; otherwise capped at
  `--recursion_steps`.
- **Optimizer**: Adam wrapped in `WarmupDecayOptimizer` (Transformer warmup + exponential
  decay), *plus* a `ReduceLROnPlateau` scheduler on the epoch loss.
- **AMP**: **off by default** (`--use_amp 0`) — frequency-domain losses are precision-sensitive
  and FP32 is more NaN-stable. When enabled on GPU, model forward runs under `autocast`, spectra
  are cast to `.float()` for the loss, and the LR is re-applied manually after `scaler.step` to
  avoid double-stepping.
- **Robustness**: NaN/Inf loss triggers an emergency checkpoint and stops; loss > 100 skips
  the batch; grads are clipped to `--max_norm`. `best_model.pth.tar` is saved on best avg loss.
  Checkpoints store `model_state_dict`, `optimizer_state`, and `criterion_state` (the ω buffer
  must be restored on resume).
  - **Known bug**: the periodic `--save_interval` save block references an undefined `path`
    variable (only `best_path` is defined), so periodic checkpoints will raise `NameError`.
    Only the best-model save currently works. Fix by defining `path` before `torch.save` if you
    rely on periodic saves.

### Inference scripts

- `test_vmd.py` — runs 10 hard-coded `TEST_SIGNAL_CONFIGS` (3–10 modes, AM/FM/CW components),
  decomposes each, prints reconstruction MSE, saves `{name}_original.png` and `{name}_modes.png`.
- `demo_vmd.py` — single 8-mode signal, fuller plots (time/freq comparisons, residual).
- `demo_lightweight.py` — self-contained train+test on a small model for quick iteration.

Recursive decomposition in the inference scripts uses `RECURSION_STEPS = 20` and a relative
residual-energy `epsilon` stop, mirroring training. Each step passes the step index into
`model(current_input, step)`. (`test_vmd.decompose` uses `epsilon=0.005`.) The scripts load
checkpoints with `strict=False` so the criterion's buffers aren't required.

### Key training CLIs

| Parameter | Default | Role |
|-----------|---------|------|
| `--recursion_steps` | 20 | Max recursion depth (hard cap) |
| `--stop_epsilon` | 0.01 | Adaptive stop: residual / original energy |
| `--weight_bandwidth` | 20.0 | `w1`: J1 weight (balanced with J2/J3; too large → mode collapse) |
| `--alpha_bandwidth` | 50.0 | `alpha`: J1 scale + β-filter sharpness in J2 (larger = narrower β) |
| `--gamma_residual` | 1.0 | `w2`: J2 (residual separation) weight |
| `--delta_history` | 2.0 | `w3`: J3 (history separation) weight |
| `--use_amp` | 0 | AMP mixed precision (0 = FP32, more NaN-stable) |
| `--n_fft` / `--hop_length` / `--hidden_dim` | 512 / 128 / 64 | Model size |

These CLI args map to `NeuralSVMDCriterion(alpha, w1, w2, w3, ...)`. There is no
reconstruction-loss weight and no energy-floor (`--eta_energy`) term — the loss is exactly
J1/J2/J3.

## Design constraints & gotchas

- **Mode/residual splitter, not an N-source separator**: the model emits one mask `M` and
  returns `mode` (`M·X`) and `residual` (`(1−M)·X`) in a dict. `forward` requires the
  recursion `step` argument — callers must pass it.
- **Reconstruction is structural, not learned** — the model emits one mask `M` and uses
  `M`/`1−M`, so `mode + residual = input` holds exactly and there is no reconstruction loss.
  Do not reintroduce independent dual masks without also restoring a reconstruction term.
- **The criterion is stateful** — `NeuralSVMDCriterion` holds the `omega` buffer; you must
  call `update_omega()` once per recursion step and save/restore `criterion_state` across
  resume, or ω resets to 0 and J1/J2 lose their reference frequency.
- **All loss terms share one (Hz) frequency axis** derived from the spectrum's bin count, so
  `alpha`/`alpha_bandwidth` has a single meaning. (A past bug mixed Hz and normalized axes,
  silently disabling the separation terms — keep the axis consistent if you edit the loss.)
- **J1/J2/J3 must stay balanced** — unlike the paper's ADMM, the neural recursive framework
  has no structural mode removal from the residual, so diversity depends on J2/J3. If
  `weight_bandwidth` is too large, the model collapses to a fixed bandpass (same center
  frequency every step). If modes bleed into each other, raise `alpha_bandwidth` instead.
- **Center frequency is always detached** in every loss term.
- **Frequency resolution** is set by `n_fft` (Δf ≈ `sample_rate / n_fft`, ≈15.6 Hz at 8 kHz
  with `n_fft=512`). Low-frequency / closely-spaced modes are limited by this — raise `n_fft`
  if low modes are poorly separated.

### Dead code (do not extend)

`transformer_improved.py`, `synthetic_data.py`, and `utils.py` are ReSepNet/DPTNet leftovers
from an earlier architecture and are **not imported anywhere** in the current pipeline. The
live model is the STFT U-Net in `models.py`. Likewise, the various `*.md` design docs in the
repo root describe earlier iterations and may not match the current code.
