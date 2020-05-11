import torch
import torch.nn as nn
import torch.nn.functional as F


class Rep(nn.Module):
    """Representation space"""

    def __init__(self):
        super(Rep, self).__init__()
        self.rep = nn.Parameter(torch.rand(2), requires_grad=True)

    def forward(self, x, rep=None):
        if rep is None:
            rep = self.rep
        else:
            rep = torch.as_tensor(rep)
        return torch.cat((x, rep))


class BasedRegressor(nn.Module):
    """ Neural network based Regressor for MAML [1]
    
    forward function has grad and learning rate, to apply
    manually updated weight to computation
    """

    def __init__(self):
        super(BasedRegressor, self).__init__()
        self.fc1 = nn.Linear(1, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x, params_dict=None, lr=0.01):
        if params_dict is None:
            params_dict = dict(self.named_parameters())
        x = F.relu(F.linear(input=x,
                            weight=params_dict['fc1.weight'],
                            bias=params_dict['fc1.bias']))
        x = F.relu(F.linear(input=x,
                            weight=params_dict['fc2.weight'],
                            bias=params_dict['fc2.bias']))
        y = F.linear(input=x,
                     weight=params_dict['fc3.weight'],
                     bias=params_dict['fc3.bias'])
        return y

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        # 64 filters, filter size 3x3
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """Use functional forward"""
        pass


def functional_forward(x, conv_weight, conv_bias, bn_weight, bn_bias):
    x = F.conv2d(input=x,
                weight=conv_weight,
                bias=conv_bias,
                stride=1, padding=1)
    x = F.batch_norm(x,
                     running_mean=None,
                     running_var=None,
                     weight=bn_weight,
                     bias=bn_bias,
                     training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2)
    return x


class MAMLImageNet(nn.Module):

    def __init__(self, n_classes, n_filters):
        super(MAMLImageNet, self).__init__()
        self.n_classes = n_classes
        self.n_filters = n_filters

        self.block1 = ConvBlock(3, n_filters)
        self.block2 = ConvBlock(n_filters, n_filters)
        self.block3 = ConvBlock(n_filters, n_filters)
        self.block4 = ConvBlock(n_filters, n_filters)
        self.fc = nn.Linear(n_filters, n_classes)

    def forward(self, x, params_dict=None):
        if params_dict is None:
            params_dict = dict(self.named_parameters())
        for i in [1,2,3,4]:
            x = functional_forward(x,
                                   params_dict[f'block{i}.conv.weight'],
                                   params_dict[f'block{i}.conv.bias'],
                                   params_dict[f'block{i}.bn.weight'],
                                   params_dict[f'block{i}.bn.bias'])
        x = x.view(x.size(0), -1)
        x = F.linear(x,
                     params_dict['fc.weight'],
                     params_dict['fc.bias'])
        return x 
