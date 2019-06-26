from torch import nn, optim
from torch.nn.functional import relu as r, softmax as s, dropout as d
from torch import tanh as ta

class Network(nn.Module):
    def __init__(self, conv1_kernel, ap1_kernel, conv2_kernel):
        super().__init__() 
        self.conv1 = nn.Conv2d(1, 6, conv1_kernel, dilation=1)
        self.ap1 = nn.MaxPool2d(ap1_kernel)
        self.conv2 = nn.Conv2d(6, 3, conv2_kernel)
        self.ln1 = nn.Linear(6615, 2025)
        self.ln2 = nn.Linear(2025, 1012)
        self.ln3 = nn.Linear(1012, 506)
        self.ln4 = nn.Linear(506, 27648)

    
    def forward(self, value, training):
        unfiltered = self.conv1(value)
        unfiltered = r(unfiltered)
        filtered = self.ap1(unfiltered)
        deep_filtered = self.conv2(filtered)
        deep_filtered = r(deep_filtered)
        value = deep_filtered
        value = value.view(-1)
        value = self.ln1(value)
        value = r(value)
        value = d(value, training=training, p=0.3)
        value = self.ln2(value)
        value = r(value)
        value = d(value, training=training, p=0.2)
        value = self.ln3(value)
        value = r(value)
        value = d(value, training=training, p=0.3)
        value = self.ln4(value)
        value = r(value)
        value = d(value, training=training, p=0.2)        
        value = value.view(1, 3, 72, 128)
        return value

#%%
