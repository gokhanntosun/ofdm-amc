import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class OFDMDataset(Dataset):
    def __init__(self, snr_db: int=None, is_training: bool=True) -> None:
        super().__init__()
        self.is_training = is_training
        self.snr_db = self.__snr_db(snr_db)
        self.xs, self.ys = self.__read_data()

    def __len__(self) -> int:
        return self.xs.shape[-1]

    def __getitem__(self, index):
        return (self.xs[:, :, index], self.ys[index, :])
    
    def __snr_db(self, snr_db: int) -> str:
        assert snr_db in list(range(-4, 28, 4)) or snr_db == NotImplemented, 'Invalid SNR value!'
        if not snr_db:
            return None
        if snr_db < 0: return f'_{-1 * snr_db}db'
        else: return f'{snr_db}db'

    def __read_data(self):
        # If change BASE to one upper directory, it will mix all data points. Currently it seperates by SNR values
        t = 'train' if self.is_training else 'test'
        BASE = f'/Users/gtosun/Documents/vsc_workspace/ofdm-amc/data/data_lib/{t}'
        if self.snr_db != None:
            BASE = os.path.join(BASE, self.snr_db)

        NUMBER_OF_ITEMS = 1000
        SYMBOL_PER_ITEM = 2048

        xpaths, ypaths = [], []
        cnt = 0
        for root, _, files in os.walk(BASE):
            for file in files:
                if file.startswith('x'): xpaths.append(os.path.join(root, file))
                if file.startswith('y'): ypaths.append(os.path.join(root, file))

        cnt = len(xpaths)
        xs = np.zeros((2, SYMBOL_PER_ITEM, NUMBER_OF_ITEMS * cnt))
        ys = np.zeros((cnt * NUMBER_OF_ITEMS, 3), dtype=np.float32)

        for xp, yp in zip(xpaths, ypaths):            
            assert os.path.exists(xp), f'Invalid path: {xp}'
            assert os.path.exists(yp), f'Invalid path: {yp}'

            x = torch.from_numpy(np.load(xp)).to(torch.float32)
            y = torch.from_numpy(np.load(yp)).to(torch.float32)

            xs[:, :, (cnt - 1) * NUMBER_OF_ITEMS: cnt * NUMBER_OF_ITEMS] = x
            ys[(cnt - 1) * NUMBER_OF_ITEMS: cnt * NUMBER_OF_ITEMS, :] = y
            cnt -= 1
            assert cnt >= 0
        assert cnt == 0
        return (xs, ys)
    
def get_dataloader(dataset: OFDMDataset, batch_size: int, shuffle: bool):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


def get_dataloaders(dataset: OFDMDataset, batch_size: int, shuffle: bool):
    train_size = int(len(dataset) * 0.85)
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset=dataset, lengths=[train_size ,test_size])

    trainloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle)
    testloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle)

    return trainloader, testloader
