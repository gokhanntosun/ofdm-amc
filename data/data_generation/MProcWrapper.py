import os
import sys
from tqdm import tqdm
import multiprocessing as mp

from .MPWrapperParams import MPWrapperParams
from .GeneratorParams import GeneratorParams
from .Generator import Generator

class MProcWrapper:
    MAX_PROCESSES = 12

    def __init__(self, params: MPWrapperParams) -> None:
        self.__generator_params_list = self.__generate_generator_params(params=params)
        self.__write_dir = params.write_dir

    def Run(self) -> None:
        self.__impl()
    
    def mp_impl(self, generator_params: GeneratorParams) -> None:
        Generator(generator_params).generate()

    def __impl(self) -> None:
        if os.path.exists(f'{self.__write_dir}/log'):
            os.remove(f'{self.__write_dir}/log')

        with mp.Pool(min(len(self.__generator_params_list), MProcWrapper.MAX_PROCESSES)) as p:
            for _ in tqdm(p.imap(self.mp_impl, self.__generator_params_list), total=len(self.__generator_params_list), colour='blue'):
                pass

    def __generate_generator_params(self, params: MPWrapperParams) -> None:
        gparams = []
        for snr in params.snr_db_list:
            for mod in params.modulation_list:
                bits_per_symbol = [2, 3] if mod == 'psk' else [4]
                for bps in bits_per_symbol:
                    for f in params.fft_size_list:
                        for channel in params.channel_list:
                            gparams.append(
                                GeneratorParams(
                                modulation=mod,
                                bits_per_symbol=bps,
                                fft_size=f,
                                channel=channel,
                                snr_db=snr,
                                write_dir=params.write_dir
                                )
                            )
        return gparams


def main():
    pass

if __name__=='__main__':
    sys.exit(main())