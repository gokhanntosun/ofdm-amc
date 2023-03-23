import numpy as np 
import numpy.fft as F

class OFDM:

    def __init__(self, n_carriers: int, cp_length: int=0) -> None:
        pass        
        self.CP_LEN = cp_length
        self.N_CARRIERS = n_carriers

    def Modulate(self, symbols: np.ndarray) -> np.ndarray:
        if self.CP_LEN >= 0:
            self.__addCyclicPrefix(symbols)
        return self.__postProcess(self.__modOFDM(self.__preProcess(symbols)))

    def Demodulate(self, symbols: np.ndarray) -> np.ndarray:
        if self.CP_LEN >= 0:
            self.__removeCyclicPrefix(symbols)
        return self.__postProcess(self.__demodOFDM(self.__preProcess(symbols)))

    def setCPLen(self, new_cp_len: int) -> None:
        self.CP_LEN = new_cp_len
        return

    def setNumberOfCarriers(self, new_n_carriers: int) -> None:
        self.N_CARRIERS = new_n_carriers
        return

    def __preProcess(self, symbols: np.ndarray) -> np.ndarray:
        return symbols[0, :] + 1j * symbols[1, :]

    def __postProcess(self, symbols: np.ndarray) -> np.ndarray:
        r = np.real(symbols)
        i = np.imag(symbols)
        return np.row_stack((r, i))

    def __modOFDM(self, symbols: np.ndarray) -> np.ndarray:
        chunks = np.reshape(symbols, (1, self.N_CARRIERS, len(symbols) // self.N_CARRIERS))
        tmp = np.zeros_like(chunks)
        for i in range(chunks.shape[-1]):
            tmp[:, :, i] = F.ifft(chunks[:, :, i], n=self.N_CARRIERS) * np.sqrt(self.N_CARRIERS)
        return np.reshape(tmp, (1, len(symbols)))

    def __demodOFDM(self, symbols: np.ndarray) -> np.ndarray:
        chunks = np.reshape(symbols, (1, self.N_CARRIERS, len(symbols) // self.N_CARRIERS))
        tmp = np.zeros_like(chunks)
        for i in range(chunks.shape[-1]):
            tmp[:, :, i] = F.fft(chunks[:, :, i], n=self.N_CARRIERS) / np.sqrt(self.N_CARRIERS)
        return np.reshape(tmp, (1, len(symbols)))

    def __addCyclicPrefix(self, frame: np.ndarray) -> np.ndarray:
        return np.concatenate((frame[-self.CP_LEN:], frame))

    def __removeCyclicPrefix(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.CP_LEN:]
