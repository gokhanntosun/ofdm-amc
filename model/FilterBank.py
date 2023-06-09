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
            #! TODO: because of the batches, you get a (2, 2048, 16) tensor. must handle this
            #* Idea: Instead of using np.fft/np.ifft, you can use torch.stft/torch.istft for OFDM (de)modulation
            chunks = np.reshape(x, (2, self.__size, x.shape[-1] // self.__size))
            output = np.empty_like(chunks)

            for i in range(chunks.shape[-1]):
                output[:, :, i] = self.__modem.demodulate(chunks[:, :, i])

            return torch.from_numpy(np.reshape(output, (2, output.size // 2))).to(torch.float32)