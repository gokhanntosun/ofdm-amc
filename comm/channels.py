import numpy as np

# TODO: Rayleigh Fading, Nakagami Fading, ISI Channels

def AWGNChannel(symbols: np.array, SNR_db: int) -> np.array:
    np.random.seed(0)
    std_dev = 1 / (2 * np.sqrt(10 ** (SNR_db / 10) ))
    n_real = np.double(std_dev * np.random.randn(np.shape(symbols)[1]))
    n_imag = np.double(std_dev * np.random.randn(np.shape(symbols)[1]))
    noise = np.row_stack((n_real, n_imag))
    return symbols + noise