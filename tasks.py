import sys
from data.data_generation import (
    MPWrapperParams,
    MProcWrapper
)

def generate_data_lib() -> None:
    mp_params = MPWrapperParams(
        channel_list    =['awgn'],
        snr_db_list     =list(range(-4, 28, 4)),
        modulation_list =['psk', 'qam'],
        fft_size_list   =[256, 512, 1024],
        write_dir       ='/Users/gtosun/Documents/vsc_workspace/ofdm-amc/data/data_lib'
        )
    MProcWrapper(mp_params).Run()

def main():
    generate_data_lib()

if __name__=='__main__':
    sys.exit(main())

