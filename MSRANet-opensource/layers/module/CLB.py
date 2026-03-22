import torch.nn as nn
import torch

class Calibration(nn.Module):
    def __init__(self, channel = 2048):
        super(Calibration, self).__init__()
        self.cali = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.w = nn.Parameter(nn.init.kaiming_normal_(torch.empty(2048)))

    def forward(self, x):
        x = self.cali(self.w*x)
        return x