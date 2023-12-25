import torch
import torch.nn as nn
from model.FilterBank import FilterBank
        
class AMCModelBaselineAlt(nn.Module):
    def __init__(self, filter_size_list: list=[256, 512, 1024]) -> None:
        super().__init__()

        self.f1 = FilterBank.FFTFilter(n=256)
        self.f2 = FilterBank.FFTFilter(n=512)
        self.f3 = FilterBank.FFTFilter(n=1024)

        self.conv1 = nn.Conv2d(1, 64, (5, 5), padding=(2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 16, (2, 2), padding=(1, 1))
        self.conv4 = nn.Conv2d(16, 1, (2, 1), padding=(1, 0))

        self.fc1 = nn.LazyLinear(out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=3)

        self.mpool1 = nn.MaxPool2d((2, 1), padding=(0, 0))
        self.mpool2 = nn.MaxPool2d((2, 2), padding=(0, 0))

        self.relu = nn.LeakyReLU(negative_slope=0.05)
        self.tanh = nn.Tanh()

    def forward(self, x):

        x1 = self.f1.filter(x)
        x2 = self.f2.filter(x)
        x3 = self.f3.filter(x)
        x = torch.row_stack((x1, x2, x3))

        x = torch.unsqueeze(torch.unsqueeze(x, 0), 0)
        x = self.mpool1( self.relu( self.conv1(x) ) )
        x = self.mpool2( self.relu( self.conv2(x) ) )
        x = self.mpool2( self.relu( self.conv3(x) ) )
        x = self.mpool2( self.relu( self.conv4(x) ) )

        x = torch.flatten(x)

        x = self.tanh( self.fc1(x) )
        x = self.tanh( self.fc2(x) )
        return torch.squeeze(self.fc3(x))
