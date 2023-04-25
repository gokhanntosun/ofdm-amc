import torch
import torch.nn as nn
import torch.nn.functional as f

from model.FilterBank import FilterBank
        
class AMCModelBaseline(nn.Module):
    def __init__(self, filter_size_list: list=[256, 512, 1024]) -> None:
        super().__init__()
        # self.__size_list = filter_size_list
        # for debugging, construct each net seperately
        # self.__cnns = [SubNetwork(fft_size=size) for size in self.__size_list]

        self.__cnn256 = AMCModelBaseline.SubNetwork(fft_size=256)
        self.__cnn512 = AMCModelBaseline.SubNetwork(fft_size=512)
        self.__cnn1024 = AMCModelBaseline.SubNetwork(fft_size=1024)
        self.__max_pool = nn.MaxPool2d((3, 1))

    def forward(self, x):
        f256 = self.__cnn256.forward(x)
        f512 = self.__cnn512.forward(x)
        f1024 = self.__cnn1024.forward(x)

        w = torch.row_stack((f256, f512, f1024))
        w = torch.unsqueeze(w, 0)
        w = self.__max_pool(w)

        return torch.squeeze(w)    # loss function already includes softmax
    
    class BaseModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, (2, 1), padding=(1, 0))
            self.conv2 = nn.Conv2d(32, 4, (2, 2), padding=(1, 1))
            self.conv3 = nn.Conv2d(4, 1, (2, 171), stride=(1, 171))
            self.max_pool= nn.MaxPool2d((2, 2), padding=(1, 1))
            self.relu = nn.LeakyReLU(negative_slope=0.05)
            

        def forward(self, x):
            x = torch.unsqueeze(x, 0)
            x = self.max_pool( self.relu( self.conv1(x) ) )
            x = self.max_pool( self.relu( self.conv2(x) ) )
            x = self.conv3(x)   # -> [1, 1, 3]

            return torch.flatten(x) # -> [1, 3]

    class SubNetwork(BaseModel, nn.Module):
        def __init__(self, fft_size: int) -> None:
            super().__init__()
            self.__filter = FilterBank.FFTFilter(n=fft_size)

        def forward(self, x):
            x = self.__filter.filter(x)
            return super().forward(x)