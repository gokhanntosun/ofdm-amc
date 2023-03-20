import os
import numpy as np
from comm.modem import (Modem, ModulationType)
from comm.ofdm import OFDM
from comm.channels import AWGNChannel
import datetime

from argparse import ArgumentParser

parser = ArgumentParser(description="Dataset generator for digital modulation classification task.")
parser.add_argument("-m", "--modulation", type=str, choices=["psk", "qam"], required=True, help="Modulation type. Example: -m qam or -m psk")
parser.add_argument("-k", "--bits_per_symbol", type=int, choices=[1, 2, 3, 4, 6], required=True, help="Bits per symbol. Must be in set \{2, 8, 16, 64\}. Example: -k 8 or -k 64")
parser.add_argument("-f", "--fft_size", type=int, choices=[256, 512, 1024], required=True)
parser.add_argument("-c", "--channel", type=str, choices=["awgn"], required=True, help="Channel type. Example: -c awgn")
parser.add_argument("-s", "--snr_db", type=int, required=False, help="Channel type. Example: -s -2")
parser.add_argument("-w", "--write_dir", type=str, required=True, help="Root directory to write the results. Required sub-directories will be created")
args = parser.parse_args()
t_start = datetime.datetime.now()

# constants
NUMBER_OF_ITEMS = 1000
INPUT_LEN = 2048

FFT_SIZE = args.fft_size
BITS_PER_SYMBOL = args.bits_per_symbol
SNR_dB = args.snr_db

SYMBOL_PER_FRAME = FFT_SIZE
FRAME_PER_SAMPLE = INPUT_LEN // SYMBOL_PER_FRAME
BITSTREAM_LEN = FRAME_PER_SAMPLE * SYMBOL_PER_FRAME * BITS_PER_SYMBOL * NUMBER_OF_ITEMS

# init 
modem = Modem(n=BITS_PER_SYMBOL, type=ModulationType.PSK)
ofdm = OFDM(n_carriers=FFT_SIZE, cp_length=0)

# bits
bitstream = np.random.randint(0, 2, (BITSTREAM_LEN))

# tinme symbols
t_symbols = modem.modulateBitstream(bitstream)

# ofdm symbols
chunks = np.reshape(t_symbols, (2, SYMBOL_PER_FRAME, FRAME_PER_SAMPLE * NUMBER_OF_ITEMS))
temp = np.empty_like(chunks)
for i in range(chunks.shape[-1]):
    temp[:, :, i] = ofdm.Modulate(chunks[:, :, i])
f_symbols = np.reshape(temp, (2, INPUT_LEN * NUMBER_OF_ITEMS))

# channel
rx_symbols = AWGNChannel(symbols=f_symbols, SNR_db=SNR_dB)

# split into samples
split = np.reshape(f_symbols, (2, INPUT_LEN, NUMBER_OF_ITEMS))

# label 
mods = ["4psk", "8psk", "16qam"]
idx = {mod: i for i, mod in enumerate(mods)}
label = np.zeros(shape=(1, len(mods)))
label[:, idx[f"{2 ** args.bits_per_symbol}{args.modulation}"]] = 1

# save symbols. write_dir is: ./data/categorical
snr = f'{SNR_dB}' if args.snr_db >= 0 else f'_{-SNR_dB}'
write_path = f"{args.write_dir}/{snr}db/{args.modulation}/{2 ** args.bits_per_symbol}"
if not os.path.exists(write_path):
    os.makedirs(write_path)

# labels = [label] * 4000
labels = [np.squeeze(label)] * NUMBER_OF_ITEMS
np.save(file=f"{write_path}/x.npy", arr=split)
np.save(file=f"{write_path}/y.npy", arr=labels)

t_end = datetime.datetime.now()
with open(f"{args.write_dir}/log", mode="a+") as w:
    w.write(f"{str(2 ** args.bits_per_symbol).rjust(2, ' ')}{args.modulation} @ {str(args.snr_db).rjust(3, ' ')}dB: START: {t_start.strftime('%Y-%m-%d %H:%M:%S')} x END: {t_end.strftime('%Y-%m-%d %H:%M:%S')} X Elapsed: {str(t_end - t_start)}\n")
