# Synthetic signal generator for Neural SVMD training
# Generates mixtures of band-limited modes — no ground truth needed

import torch
import numpy as np


def generate_mode(T, sample_rate, center_freq, bandwidth, device='cpu'):
    """
    Generate a single band-limited mode via Gaussian spectrum + random phase.

    Args:
        T: signal length in samples
        sample_rate: sample rate (Hz)
        center_freq: center frequency of the mode (Hz)
        bandwidth: standard deviation of the Gaussian spectrum (Hz)
    Returns:
        mode: [T] normalized mode signal
    """
    n_fft = T // 2 + 1
    freqs = torch.linspace(0, sample_rate / 2, n_fft, device=device)

    # Gaussian spectral envelope
    envelope = torch.exp(-0.5 * ((freqs - center_freq) / (bandwidth + 1e-8)) ** 2)

    # Random phase per frequency bin
    phase = torch.rand(n_fft, device=device) * 2 * np.pi
    complex_spec = envelope * torch.exp(1j * phase)

    # IFFT back to time domain
    mode = torch.fft.irfft(complex_spec, n=T)

    # Normalize to unit variance
    mode = mode / (mode.std() + 1e-8)
    return mode


def generate_mixture(batch_size, T, sample_rate,
                     n_modes_min=2, n_modes_max=5,
                     freq_range=(100, 3800),
                     bw_range=(20, 150),
                     device='cpu'):
    """
    Generate a batch of synthetic mixtures.

    Each mixture is a sum of several band-limited modes with random center
    frequencies, bandwidths, and amplitudes.

    Args:
        batch_size: number of mixtures to generate
        T: signal length in samples
        sample_rate: sample rate (Hz)
        n_modes_min, n_modes_max: range for number of modes per mixture
        freq_range: (min, max) center frequency range (Hz)
        bw_range: (min, max) bandwidth range (Hz)
    Returns:
        mixtures: [B, T]
        n_modes: [B] actual number of modes per sample
    """
    mixtures = torch.zeros(batch_size, T, device=device)
    n_modes_list = []

    for b in range(batch_size):
        n_modes = np.random.randint(n_modes_min, n_modes_max + 1)
        n_modes_list.append(n_modes)
        mixture = torch.zeros(T, device=device)

        for _ in range(n_modes):
            fc = (torch.rand(1).item() *
                  (freq_range[1] - freq_range[0]) + freq_range[0])
            bw = (torch.rand(1).item() *
                  (bw_range[1] - bw_range[0]) + bw_range[0])
            mode = generate_mode(T, sample_rate, fc, bw, device)
            amp = torch.rand(1).item() * 0.5 + 0.5
            mixture += amp * mode

        mixtures[b] = mixture

    return mixtures, torch.tensor(n_modes_list)


class SyntheticDataset:
    """
    Simple iterable that generates fresh random mixtures each epoch.

    Since data is synthetic and infinite, each call to __iter__ produces
    `num_batches` worth of fresh random mixtures.
    """

    def __init__(self, batch_size, T, sample_rate,
                 num_batches=200,
                 n_modes_min=2, n_modes_max=5,
                 freq_range=(100, 3800),
                 bw_range=(20, 150),
                 device='cpu'):
        self.batch_size = batch_size
        self.T = T
        self.sample_rate = sample_rate
        self.num_batches = num_batches
        self.n_modes_min = n_modes_min
        self.n_modes_max = n_modes_max
        self.freq_range = freq_range
        self.bw_range = bw_range
        self.device = device

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            mixtures, n_modes = generate_mixture(
                self.batch_size, self.T, self.sample_rate,
                self.n_modes_min, self.n_modes_max,
                self.freq_range, self.bw_range,
                self.device)

            lengths = torch.full((self.batch_size,), self.T, dtype=torch.long)

            # Return format matches what solver expects:
            # (padded_mixture, mixture_lengths, _, _)
            yield mixtures, lengths, None, None
