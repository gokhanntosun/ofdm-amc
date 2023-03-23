import os
import sys

import datetime
import numpy as np
from argparse import ArgumentParser

from .GeneratorParams import GeneratorParams
from comm import (
    Modem, ModulationType,
    # Channel, ChannelType,
    OFDM,
    )

class Generator:

    # static constants
    NUMBER_OF_ITEMS = 1000
    INPUT_LEN = 2048

    def __init__(self, params: GeneratorParams) -> None:
        # comm parameters
        self.__FFT_SIZE = params.fft_size
        self.__BITS_PER_SYMBOL = params.bits_per_symbol
        self.__SNR_dB = params.snr_db

        # data structure
        self.__SYMBOL_PER_FRAME = self.__FFT_SIZE
        self.__FRAME_PER_SAMPLE = Generator.INPUT_LEN // self.__SYMBOL_PER_FRAME
        self.__BITSTREAM_LEN = self.__FRAME_PER_SAMPLE * self.__SYMBOL_PER_FRAME * self.__BITS_PER_SYMBOL * Generator.NUMBER_OF_ITEMS

        # init 
        self.__WRITE_DIR = params.write_dir
        self.__MOD_TYPE_STR = params.modulation
        self.__MOD_TYPE = ModulationType.PSK if self.__MOD_TYPE_STR == 'psk' else ModulationType.QAM
        self.__modem = Modem(n=self.__BITS_PER_SYMBOL, type=self.__MOD_TYPE)
        self.__ofdm = OFDM(n_carriers=self.__FFT_SIZE, cp_length=0)
        # self.__channel = Channel(type=ChannelType.AWGN, SNR_db=self.__SNR_dB)

    def generate(self) -> None:
        t_start = datetime.datetime.now()
        self.__impl()
        t_end = datetime.datetime.now()
        with open(f"{self.__WRITE_DIR}/log", mode="a+") as w:
            w.write(
            f'{str(2 ** self.__BITS_PER_SYMBOL).rjust(2, " ")}'
            f'{self.__MOD_TYPE_STR} - {str(self.__FFT_SIZE).rjust(4, " ")}POINT'
            f' @ {str(self.__SNR_dB).rjust(3, " ")}dB: '
            f'START: {t_start.strftime("%Y-%m-%d %H:%M:%S")} x '
            f'END: {t_end.strftime("%Y-%m-%d %H:%M:%S")} X '
            f'Elapsed: {str(t_end - t_start)}\n')
        

    def __impl(self) -> None:
        self.__postprocess(self.__comm_cycle())
    
    def __comm_cycle(self) -> np.ndarray:
        #! TODO: use awgn channel output when model is ok
        bitstream = np.random.randint(0, 2, (self.__BITSTREAM_LEN))     # bits
        t_symbols = self.__modem.modulateBitstream(bitstream)           # time symbols
        f_symbols = self.__ofdm.Modulate(t_symbols)
        # rx_symbols = self.__channel.apply(f_symbols)  # awgn channel
        return f_symbols

    def __postprocess(self, rx_symbols: np.ndarray) -> None:
        split = np.reshape(rx_symbols, (2, Generator.INPUT_LEN, Generator.NUMBER_OF_ITEMS))

        mods = ["4psk", "8psk", "16qam"]                # labels
        idx = {mod: i for i, mod in enumerate(mods)}    # label indices
        label = np.zeros(shape=(1, len(mods)))
        label[:, idx[f"{2 ** self.__BITS_PER_SYMBOL}{self.__MOD_TYPE_STR}"]] = 1

        snr = f'{self.__SNR_dB}' if self.__SNR_dB >= 0 else f'_{-self.__SNR_dB}'
        write_path = f"{self.__WRITE_DIR}/{snr}db/{self.__MOD_TYPE_STR}/{2 ** self.__BITS_PER_SYMBOL}/{self.__FFT_SIZE}point"
        os.makedirs(write_path, exist_ok=True)  # write to file

        labels = [np.squeeze(label)] * Generator.NUMBER_OF_ITEMS
        np.save(file=f"{write_path}/x.npy", arr=split)
        np.save(file=f"{write_path}/y.npy", arr=labels)


def main():

    parser = ArgumentParser(description="Dataset generator for digital modulation classification task.")
    parser.add_argument("-m", "--modulation", type=str, choices=["psk", "qam"], required=True, help="Modulation type. Example: -m qam or -m psk")
    parser.add_argument("-k", "--bits_per_symbol", type=int, choices=[1, 2, 3, 4, 6], required=True, help="Bits per symbol. Must be in set \{2, 8, 16, 64\}. Example: -k 8 or -k 64")
    parser.add_argument("-f", "--fft_size", type=int, choices=[256, 512, 1024], required=True)
    parser.add_argument("-c", "--channel", type=str, choices=["awgn"], required=True, help="Channel type. Example: -c awgn")
    parser.add_argument("-s", "--snr_db", type=int, required=False, help="Channel type. Example: -s -2")
    parser.add_argument("-w", "--write_dir", type=str, required=True, help="Root directory to write the results. Required sub-directories will be created")
    args = parser.parse_args()

    params = GeneratorParams(
        modulation = args.modulation,
        bits_per_symbol = args.bits_per_symbol,
        fft_size = args.fft_size,
        channel = args.channel,
        snr_db = args.snr_db,
        write_dir = args.write_dir,
    )

    gen = Generator(params=params)
    gen.generate()

    return

if __name__=='__main__':
    sys.exit(main())