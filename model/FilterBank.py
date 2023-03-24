import torch
import numpy as np
from comm.OFDM import OFDM
from typing import List

class FilterBank:
    SIZE_LIST = [256, 512, 1024]

    def __init__(self) -> None:
        self.__filters = self.__generate_filters()

    def __generate_filters(self) -> List:
        return [FilterBank.FFTFilter(size) for size in FilterBank.SIZE_LIST]

    def filter(self, x) -> torch.tensor:
        return torch.tensor(np.array([filter.filter(x) for filter in self.__filters]))

    class FFTFilter:
        def __init__(self, n: int) -> None:
            self.__size = n
            self.__modem = OFDM(n_carriers=n, cp_length=0)

        def filter(self, x) -> np.ndarray:
            chunks = np.reshape(x, (2, self.__size, x.shape[-1] // self.__size))
            output = np.empty_like(chunks)

            for i in range(chunks.shape[-1]):
                output[:, :, i] = self.__modem.demodulate(chunks[:, :, i])

            return np.reshape(output, (2, output.size // 2))