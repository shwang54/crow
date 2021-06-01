import warnings

import torch.nn as nn
import torch.nn.functional as F


class F_conv(nn.Module):
    '''ResNet transformation, not itself reversible, just used below'''

    def __init__(self, in_channels, channels, channels_hidden=None,
                 stride=None, kernel_size=3, leaky_slope=0.1,
                 batch_norm=False):
        super(F_conv, self).__init__()

        if stride:
            warnings.warn("Stride doesn't do anything, the argument should be "
                          "removed", DeprecationWarning)
        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.conv1 = nn.Conv2d(in_channels, channels_hidden,
                               kernel_size=kernel_size, padding=pad,
                               bias=not batch_norm)
        self.conv2 = nn.Conv2d(channels_hidden, channels_hidden,
                               kernel_size=kernel_size, padding=pad,
                               bias=not batch_norm)
        self.conv3 = nn.Conv2d(channels_hidden, channels,
                               kernel_size=kernel_size, padding=pad,
                               bias=not batch_norm)

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(channels_hidden)
            self.bn1.weight.data.fill_(1)
            self.bn2 = nn.BatchNorm2d(channels_hidden)
            self.bn2.weight.data.fill_(1)
            self.bn3 = nn.BatchNorm2d(channels)
            self.bn3.weight.data.fill_(1)
        self.batch_norm = batch_norm

    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = F.leaky_relu(out, self.leaky_slope)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = F.leaky_relu(out, self.leaky_slope)

        out = self.conv3(out)
        if self.batch_norm:
            out = self.bn3(out)
        return out


class F_fully_connected(nn.Module):
    '''Fully connected tranformation, not reversible, but used below.'''

    def __init__(self, size_in, size, internal_size=None, dropout=0.0,
                 batch_norm=False):
        super(F_fully_connected, self).__init__()
        if not internal_size:
            internal_size = 2*size

        self.d1 = nn.Dropout(p=dropout)
        self.d2 = nn.Dropout(p=dropout)
        self.d2b = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(size_in, internal_size)
        self.fc2 = nn.Linear(internal_size, internal_size)
        self.fc2b = nn.Linear(internal_size, internal_size)
        self.fc3 = nn.Linear(internal_size, size)

        self.nl1 = nn.ReLU()
        self.nl2 = nn.ReLU()
        self.nl2b = nn.ReLU()

        if batch_norm:
            self.bn1 = nn.BatchNorm1d(internal_size)
            self.bn1.weight.data.fill_(1)
            self.bn2 = nn.BatchNorm1d(internal_size)
            self.bn2.weight.data.fill_(1)
            self.bn2b = nn.BatchNorm1d(internal_size)
            self.bn2b.weight.data.fill_(1)
        self.batch_norm = batch_norm

    def forward(self, x):
        out = self.fc1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.nl1(self.d1(out))

        out = self.fc2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out = self.nl2(self.d2(out))

        out = self.fc2b(out)
        if self.batch_norm:
            out = self.bn2b(out)
        out = self.nl2b(self.d2b(out))

        out = self.fc3(out)
        return out
    
class F_GRU(nn.Module):
    def __init__(self, size_in, size, internal_size=None, hidden_size=None, dropout=0.0, batch_norm=False):
        super(F_GRU, self).__init__()
        if not internal_size:
            internal_size = 2*size
        
        if not hidden_size:
            hidden_size = size_in

        self.gru1 = nn.GRUCell(size_in, hidden_size)
        self.fc1 = nn.Linear(hidden_size, internal_size)
        self.fc2 = nn.Linear(internal_size, internal_size)
        self.fc3 = nn.Linear(internal_size, size)
        self.nl1 = nn.ReLU()
        self.nl2 = nn.ReLU()
        self.sig2 = nn.Sigmoid()

    def forward(self, x, h):
#         print("x.is_cuda: ", x.is_cuda)
#         print("h.is_cuda: ", h.is_cuda)
        # GRU(x,h) -> h
        h_t = self.gru1(x, h)
        
        # FC(h) -> x
        out = self.fc1(h_t)
        out = self.nl1(out)
        
        out = self.fc2(out)
        out = self.nl2(out)
        
        out = self.fc3(out)
        return out, h_t
    
class F_GRU2(nn.Module):
    def __init__(self, size_in, size, internal_size=None, hidden_size=None, dropout=0.0, batch_norm=False):
        super(F_GRU2, self).__init__()
        if not internal_size:
            internal_size = 2*size
        
        if not hidden_size:
            hidden_size = size_in

        self.fc1 = nn.Linear(size_in, size_in)
        self.nl1 = nn.ReLU()
        self.gru1 = nn.GRUCell(size_in, hidden_size)
        self.fc2 = nn.Linear(hidden_size, internal_size)
        self.nl2 = nn.ReLU()
        self.fc3 = nn.Linear(internal_size, size)
        
        self.bn1 = nn.BatchNorm1d(size_in)
        self.bn1.weight.data.fill_(1)
        
        self.bn2 = nn.BatchNorm1d(internal_size)
        self.bn2.weight.data.fill_(1)

    def forward(self, x, h):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.nl1(out)
        
        # GRU(x,h) -> h
        h_t = self.gru1(out, h)
        
        out = self.fc2(h_t)
        out = self.bn2(out)
        out = self.nl2(out)
        
        out = self.fc3(out)
        return out, h_t

    
class F_GRU_cgate(nn.Module):
    def __init__(self, size_in, size, internal_size=None, hidden_size=None, dropout=0.0, batch_norm=False):
        super(F_GRU_cgate, self).__init__()
        if not internal_size:
            internal_size = 2*size
        
        if not hidden_size:
            hidden_size = size_in
            
        self.fc0 = nn.Linear(size_in, internal_size)
        self.nl0 = nn.ReLU()

        self.fc1 = nn.Linear(internal_size, internal_size)
        self.sig1 = nn.Sigmoid()
        
        self.gru1 = nn.GRUCell(internal_size, hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, internal_size)
        self.nl2 = nn.ReLU()
        
        self.fc3 = nn.Linear(internal_size, internal_size)
        self.sig3 = nn.Sigmoid()
        
        self.fc4 = nn.Linear(internal_size, size)

    def forward(self, x, h):
#         print("x.is_cuda: ", x.is_cuda)
#         print("h.is_cuda: ", h.is_cuda)

        out = self.nl0(self.fc0(x))
        
        out_fc1 = self.sig1(self.fc1(out))
        out = out * out_fc1
        
        # GRU(x,h) -> h
        h_t = self.gru1(out, h)
        
        out = self.nl2((self.fc2(h_t)))
        
        out_fc3 = self.sig3(self.fc3(out))
        out = out * out_fc3
        
        out = self.fc4(out)
        
        return out, h_t

class F_cgate(nn.Module):
    def __init__(self, size_in, dropout=0.0, batch_norm=False):
        super(F_cgate, self).__init__()
            
        self.fc0 = nn.Linear(size_in, size_in)
#         self.sig0 = nn.Sigmoid()
        self.relu0 = nn.ReLU()
        self.sig0 = nn.LogSigmoid()

    def forward(self, x):
        out = self.sig0(self.relu0(self.fc0(x+2)))
        
        return out