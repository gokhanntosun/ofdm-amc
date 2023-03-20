import os
import subprocess as sp
import multiprocessing as mp
from tqdm import tqdm

def run(cmd: str):
    p = sp.Popen(cmd, shell=True)
    p.communicate()

if __name__=='__main__':
    channels = ['awgn']
    snr_db = list(range(-4, 28, 4))
    modulations = ['psk', 'qam']
    fft_sizes = [256, 512, 1024]
    write_dir = '/Users/gtosun/Documents/vsc_workspace/ofdm-amc/data/data_lib'
    generator_path = '/Users/gtosun/Documents/vsc_workspace/ofdm-amc/data/data_generation/generator.py'

    # generate commands
    commands = []
    for snr in snr_db:
        for mod in modulations:
            bits_per_symbol = [2, 3] if mod == 'psk' else [4]
            for bps in bits_per_symbol:
                for f in fft_sizes:
                    commands.append(f'python3 -B {generator_path} -m {mod} -k {bps} -f {f} -c {channels[0]} -s {snr} -w {write_dir}')

    if os.path.exists(f'{write_dir}/log'):
        os.remove(f'{write_dir}/log')
    
    with mp.Pool(processes=8) as p:
        for _ in tqdm(p.imap(run, commands), total=len(commands), colour='blue'):
            pass