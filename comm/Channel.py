import numpy as np
from enum import Enum

class ChannelType(Enum):
    AWGN = 1,
    Rayleigh = 2

class Channel:
    def __init__(self, type: ChannelType, SNR_db: int=20) -> None:
        self.__type = type
        self.__SNR_db = SNR_db

    def apply(self, symbols: np.ndarray) -> np.ndarray:
        if self.__type == ChannelType.AWGN:
            return self.__awgn(symbols, self.__SNR_db)        
        elif self.__type == ChannelType.Rayleigh:
            pass

    def __awgn(self, symbols: np.ndarray, SNR_db: int) -> np.array:
        np.random.seed(0)
        std_dev = 1 / (2 * np.sqrt(10 ** (SNR_db / 10) ))
        n_real = np.double(std_dev * np.random.randn(np.shape(symbols)[1]))
        n_imag = np.double(std_dev * np.random.randn(np.shape(symbols)[1]))
        noise = np.row_stack((n_real, n_imag))
        return symbols + noise