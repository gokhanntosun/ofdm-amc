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

        self.__fc1 = nn.Linear(in_features=195, out_features=128)
        self.__fc2 = nn.Linear(in_features=128, out_features=64)
        self.__fc3 = nn.Linear(in_features=64, out_features=3)
        self.__smax = nn.Softmax(dim=0)

    def forward(self, x):
        f256 = self.__cnn256.forward(x)
        f512 = self.__cnn512.forward(x)
        f1024 = self.__cnn1024.forward(x)
        w = torch.cat([f256, f512, f1024])

        w = self.__fc1(w)
        w = self.__fc2(w)
        w = self.__fc3(w)

        return self.__smax(w)
    
    class BaseModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.conv1 = nn.Conv2d(1, 256, (2, 3), padding=(1, 1))
            self.conv2 = nn.Conv2d(256, 128, (1, 3), padding=(0, 1))
            self.conv3 = nn.Conv2d(128, 64, (1, 3), padding=(0, 1))
            self.conv4 = nn.Conv2d(64, 32, (1, 5), padding=(0, 2))
            self.conv5 = nn.Conv2d(32, 16, (1, 7), padding=(0, 3))

            self.max_pool = nn.MaxPool2d((1, 2), padding=(0, 1))
            self.max_pool_3d = nn.MaxPool3d((16, 2, 1))

            self.relu = nn.ReLU()
            

        def forward(self, x):
            x = self.max_pool( self.relu( self.conv1(x) ) )
            x = self.max_pool( self.relu( self.conv2(x) ) )
            x = self.max_pool( self.relu( self.conv3(x) ) )
            x = self.max_pool( self.relu( self.conv4(x) ) )
            return x

    class SubNetwork(BaseModel, nn.Module):
        def __init__(self, fft_size: int) -> None:
            super().__init__()
            self.__filter = FilterBank.FFTFilter(n=fft_size)

        def forward(self, x):
            x = self.__filter.filter(x)
            return super().forward(x)