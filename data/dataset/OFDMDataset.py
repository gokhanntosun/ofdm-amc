import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class OFDMDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.xs, self.ys = self.__read_data()

    def __len__(self) -> int:
        return self.xs.shape[-1]

    def __getitem__(self, index):
        return (self.xs[:, :, index], self.ys[index])

    def __read_data(self):
        # If change BASE to one upper directory, it will mix all data points. Currently it seperates by SNR values
        BASE = '/Users/gtosun/Documents/vsc_workspace/ofdm-amc/data/data_lib'

        NUMBER_OF_ITEMS = 1000
        SYMBOL_PER_ITEM = 2048

        # could not broadcast input array from shape (2,2048,1000) into shape (1000,2,2048)

        xpaths, ypaths = [], []
        cnt = 0
        for root, _, files in os.walk(BASE):
            for file in files:
                if file.startswith('x'): xpaths.append(os.path.join(root, file))
                if file.startswith('y'): ypaths.append(os.path.join(root, file))

        cnt = len(xpaths)
        # xs = np.zeros((NUMBER_OF_ITEMS * cnt, 2, SYMBOL_PER_ITEM))
        xs = np.zeros((2, SYMBOL_PER_ITEM, NUMBER_OF_ITEMS * cnt))
        ys = [""] * (cnt * NUMBER_OF_ITEMS)

        for xp, yp in zip(xpaths, ypaths):            
            assert os.path.exists(xp), f'Invalid path: {xp}'
            assert os.path.exists(yp), f'Invalid path: {yp}'

            x = torch.from_numpy(np.load(xp)).double()
            y = np.load(yp)

            xs[:, :, (cnt - 1) * NUMBER_OF_ITEMS: cnt * NUMBER_OF_ITEMS] = x.double()
            ys[(cnt - 1) * NUMBER_OF_ITEMS: cnt * NUMBER_OF_ITEMS] = y
            cnt -= 1
            assert cnt >= 0
        assert cnt == 0
        return (xs, ys)

def get_dataloaders(dataset: OFDMDataset, batch_size: int, shuffle: bool):
    train_size = int(len(dataset) * 0.85)
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset=dataset, lengths=[train_size ,test_size], generator=torch.Generator().manual_seed(0))

    trainloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle)
    testloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle)

    return trainloader, testloader
