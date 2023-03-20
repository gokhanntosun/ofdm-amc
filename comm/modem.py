from enum import Enum
from typing import List 
import numpy as np

# TODO: Add Gray Coding for improved SNR

class ModulationType(Enum):
    PSK = 1,
    QAM = 2

class Modem():
    def __init__(self, n: int, type: ModulationType):
        self.setN(n=n)
        self.setType(type=type)
        self.symbols = self.__updateSymbols()

    def setN(self, n: int) -> None:
        assert isinstance(n, int), "n must be an int."
        self.n = n

    def setType(self, type: ModulationType) -> None:
        assert isinstance(type, ModulationType), "type must be a ModulationType."
        self.type = type
        self.symbols = self.__updateSymbols()

    def modulateBitstream(self, bits: np.ndarray):
        l = len(bits)

        nrows = int(l / self.n)
        ncols = int(self.n)
        bits = np.reshape(bits, newshape=(nrows, ncols))
        bits = [''.join(seq.astype(str)) for seq in bits]

        if self.type == ModulationType.PSK:
            return self.__modPSK(bits=bits)
        
        elif self.type == ModulationType.QAM:
            return self.__modQAM(bits=bits)
            
        else:
            pass
            # raise Exception()

    def demodulateSymbolStream(self, symbols: np.array):
        # TODO: You can merge different __demod methods, they practically do the same thing
        symbols = symbols[0, :] + 1j * symbols[1, :]
        if self.type == ModulationType.PSK:
            return self.__demodPSK(symbols=symbols)

        elif self.type == ModulationType.QAM:
            return self.__demodQAM(symbols=symbols)

        else:
            pass

    def __modPSK(self, bits: np.array) -> np.ndarray:

        k = self.n      # Bits per symbol
        M = 2 ** k      # Number of symbols

        seq_to_symbol_map = {format(i, 'b').rjust(k, '0'): self.symbols[i] for i in range(M)}
        sym_seq = [seq_to_symbol_map[seq] for seq in bits]

        I = np.real(sym_seq)    # In-phase component
        Q = np.imag(sym_seq)    # Quadrature component

        return np.row_stack((I, Q))

    def __demodPSK(self, symbols: np.array):
        symbol_to_seq_map = {self.symbols[i]: format(i, 'b').rjust(self.n, '0') for i in range(2 ** self.n)}
        decoded_bits = [""] * len(symbols)  

        # Maximum likelihood decoding
        for i, symbol in enumerate(symbols):
            decoded_bits[i] = symbol_to_seq_map[self.symbols[np.argmin(np.abs(self.symbols - symbol))]]

        decoded_bits = np.array([int(bit) for bit in ''.join(decoded_bits)])
        return decoded_bits

    def __modQAM(self, bits: np.ndarray):
        k = self.n
        M = 2 ** k

        seq_to_sym_map = {format(i, 'b').rjust(k, '0'): self.symbols[i] for i in range(M)}
        sym_seq = [seq_to_sym_map[seq] for seq in bits]

        I = np.real(sym_seq)
        Q = np.imag(sym_seq)

        return np.row_stack((I, Q))

    def __demodQAM(self, symbols: np.array):
        symbol_to_seq_map = {self.symbols[i]: format(i, 'b').rjust(self.n, '0') for i in range(2 ** self.n)}
        decoded_bits = [""] * len(symbols)

        for i, symbol in enumerate(symbols):
            decoded_bits[i] = symbol_to_seq_map[self.symbols[np.argmin(np.abs(self.symbols - symbol))]]

        decoded_bits = np.array([int(bit) for bit in ''.join(decoded_bits)])
        return decoded_bits

    def __updateSymbols(self) -> np.ndarray:
        if self.type == ModulationType.PSK:
            # shift = 1.0 if self.n == 1 else np.exp(1j * np.pi / 4)
            return np.exp(1j * 2 * np.pi * np.arange(2 ** self.n) / 2 ** self.n)
        elif self.type == ModulationType.QAM:
            Ar = np.arange(start=-np.sqrt(2 ** self.n)+1, stop=np.sqrt(2 ** self.n), step=2)
            Ai = 1j * Ar
            symbols = np.array([real + imag for real in Ar for imag in Ai])
            symbols /= np.sqrt(2 * (2 ** self.n - 1) / 3)
            return symbols
        else:
            pass

    def __updateMaps(self):
        #TODO: Update symbol maps when modulation scheme or bits per symbol changes
        pass

    def __grayCode(self, n: int) -> List[str]: 
        b = [0, 1] 
        for _ in range(n-1): 
            u = b + list(reversed(b)) 
            w = [0] * int(len(u) / 2) + [1] * int(len(u) / 2) 
            b = [f"{ww}{uu}" for ww, uu in zip(w, u)] 
        return b