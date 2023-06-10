import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from model.AMCModel import AMCModel
from data.dataset.OFDMDataset import OFDMDataset, get_dataloaders
from data.data_generation import (
    MPWrapperParams,
    MProcWrapper
    )

def generate_data_lib(is_training: bool=True) -> None:
    t = 'train' if is_training else 'test'
    mp_params = MPWrapperParams(
        channel_list    =['awgn'],
        snr_db_list     =list(range(-4, 28, 4)),
        modulation_list =['psk', 'qam'],
        fft_size_list   =[256, 512, 1024],
        write_dir       =f'/Users/gtosun/Documents/vsc_workspace/ofdm-amc/data/data_lib/{t}'
        )
    MProcWrapper(mp_params).Run()

def test_models() -> None:
    pass

def train_main_model() -> None:
    pass

def train_baseline_model() -> None:
    pass

def main():
    generate_data_lib(is_training=False)

if __name__=='__main__':
    sys.exit(main())

