class GeneratorParams:
    # TODO: Add types
    def __init__(
            self,
            modulation,
            bits_per_symbol,
            fft_size,
            channel,
            snr_db,
            write_dir,
            ) -> None:
        self.modulation = modulation
        self.bits_per_symbol = bits_per_symbol
        self.fft_size = fft_size
        self.channel = channel
        self.snr_db = snr_db
        self.write_dir = write_dir