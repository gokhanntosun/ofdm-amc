import numpy as np 
import numpy.fft as F

class OFDM:

    def __init__(self, n_carriers: int, cp_length: int=0) -> None:
        pass        
        self.CP_LEN = cp_length
        self.N_CARRIERS = n_carriers

    def modulate(self, symbols: np.ndarray) -> np.ndarray:
        if self.CP_LEN >= 0:
            self.__add_cyclic_prefix(symbols)
        return self.__postprocess(self.__mod_ofdm(self.__preprocess(symbols)))

    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        if self.CP_LEN >= 0:
            self.__remove_cyclic_prefix(symbols)
        return self.__postprocess(self.__demod_ofdm(self.__preprocess(symbols)))

    def set_cp_len(self, new_cp_len: int) -> None:
        self.CP_LEN = new_cp_len
        return

    def set_number_of_carriers(self, new_n_carriers: int) -> None:
        self.N_CARRIERS = new_n_carriers
        return

    def __preprocess(self, symbols: np.ndarray) -> np.ndarray:
        return symbols[0, :] + 1j * symbols[1, :]

    def __postprocess(self, symbols: np.ndarray) -> np.ndarray:
        r = np.real(symbols)
        i = np.imag(symbols)
        return np.row_stack((r, i))

    def __mod_ofdm(self, symbols: np.ndarray) -> np.ndarray:
        chunks = np.reshape(symbols, (1, self.N_CARRIERS, len(symbols) // self.N_CARRIERS))
        tmp = np.zeros_like(chunks)
        for i in range(chunks.shape[-1]):
            tmp[:, :, i] = F.ifft(chunks[:, :, i], n=self.N_CARRIERS) * np.sqrt(self.N_CARRIERS)
        return np.reshape(tmp, (1, len(symbols)))

    def __demod_ofdm(self, symbols: np.ndarray) -> np.ndarray:
        chunks = np.reshape(symbols, (1, self.N_CARRIERS, len(symbols) // self.N_CARRIERS))
        tmp = np.zeros_like(chunks)
        for i in range(chunks.shape[-1]):
            tmp[:, :, i] = F.fft(chunks[:, :, i], n=self.N_CARRIERS) / np.sqrt(self.N_CARRIERS)
        return np.reshape(tmp, (1, len(symbols)))

    def __add_cyclic_prefix(self, frame: np.ndarray) -> np.ndarray:
        return np.concatenate((frame[-self.CP_LEN:], frame))

    def __remove_cyclic_prefix(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.CP_LEN:]
