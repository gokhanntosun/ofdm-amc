from typing import List

class MPWrapperParams:
    def __init__(
            self,
            channel_list: List[str],
            snr_db_list: List[int],
            modulation_list: List[str],
            fft_size_list: List[int],
            write_dir: str,
            ) -> None:
        self.channel_list = channel_list
        self.snr_db_list = snr_db_list
        self.modulation_list = modulation_list
        self.fft_size_list = fft_size_list
        self.write_dir = write_dir