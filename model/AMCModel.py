import torch
import torch.nn as nn
import torch.nn.functional as f

from .FilterBank import FilterBank
        
class AMCModel(nn.Module):
    def __init__(self, filter_size_list: list=[256, 512, 1024]) -> None:
        super().__init__()
        # self.__size_list = filter_size_list
        # for debugging, construct each net seperately
        # self.__cnns = [SubNetwork(fft_size=size) for size in self.__size_list]

        self.__cnn256 = AMCModel.SubNetwork(fft_size=256)
        self.__cnn512 = AMCModel.SubNetwork(fft_size=512)
        self.__cnn1024 = AMCModel.SubNetwork(fft_size=1024)

        self.__fc1 = nn.LazyLinear(out_features=128)
        self.__fc2 = nn.Linear(in_features=128, out_features=64)
        self.__fc3 = nn.Linear(in_features=64, out_features=3)

        self.__tanh = nn.Tanh()

    def forward(self, x):
        f256 = self.__cnn256.forward(x)
        f512 = self.__cnn512.forward(x)
        f1024 = self.__cnn1024.forward(x)
        w = torch.cat([f256, f512, f1024])

        w = self.__tanh(self.__fc1(w))
        w = self.__tanh(self.__fc2(w))
        return self.__fc3(w)
    
    class BaseModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, (2, 1), padding=(1, 0))
            self.conv2 = nn.Conv2d(32, 4, (2, 2), padding=(1, 1))
            self.max_pool= nn.MaxPool2d((2, 2), padding=(1, 1))
            self.relu = nn.LeakyReLU(negative_slope=0.05)
            

        def forward(self, x):
            x = torch.unsqueeze(x, 0)
            x = self.max_pool( self.relu( self.conv1(x) ) )
            x = self.max_pool( self.relu( self.conv2(x) ) )

            return torch.flatten(x)

    class SubNetwork(BaseModel, nn.Module):
        def __init__(self, fft_size: int) -> None:
            super().__init__()
            self.__filter = FilterBank.FFTFilter(n=fft_size)

        def forward(self, x):
            # consider adding normalization after filtering
            x = self.__filter.filter(x)
            return super().forward(x)