import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
    """ MAML base learner for mini ImageNet dataset,
    Each image is 84x84, 3 channels.
    For n_filter =64, final flatten feature map is 1600
    """

    def __init__(self, n_classes, n_filters: int = 64, H_in: int = 84):
        super(MAMLImageNet, self).__init__()
        self.n_classes = n_classes
        self.n_filters = n_filters
        # Calculate H_out after 4 ConvBlocks
        for i in range(4):
            H_in = int(np.floor(H_in/2))

        self.block1 = ConvBlock(3, n_filters)
        self.block2 = ConvBlock(n_filters, n_filters)
        self.block3 = ConvBlock(n_filters, n_filters)
        self.block4 = ConvBlock(n_filters, n_filters)
        self.fc = nn.Linear(n_filters*H_in*H_in, n_classes)

    def forward(self, x, params_dict=None):
        if params_dict is None:
            params_dict = dict(self.named_parameters())
        for i in [1, 2, 3, 4]:
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


class MAMLOmniglot(nn.Module):
    """ MAML based learner for Omniglot dataset"""

    def __init__(self, n_classes, n_filters: int = 64, H_in: int = 26):
        super(MAMLOmniglot, self).__init__()
        self.n_classes = n_classes
        self.n_filters = 64
        # Calculate H_out after 4 ConvBlocks
        for i in range(4):
            H_in = int(np.floor(H_in/2))

        self.block1 = ConvBlock(1, n_filters)
        self.block2 = ConvBlock(n_filters, n_filters)
        self.block3 = ConvBlock(n_filters, n_filters)
        self.block4 = ConvBlock(n_filters, n_filters)
        self.fc = nn.Linear(n_filters*H_in*H_in, n_classes)

    def forward(self, x, params_dict=None):
        if params_dict is None:
            params_dict = dict(self.named_parameters())
        for i in [1, 2, 3, 4]:
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


class MAMLConv(nn.Module):
    """ MAML based learner for both Omniglot and miniImageNet dataset
    n_way: number of classes in n_way k_shot classification
    C_in: number of input channels
    H_in: height of input image (here input images are square with H_in = W_in)
    """

    def __init__(self,
                 n_way,
                 n_filters: int = 64,
                 C_in: int = 3,
                 H_in: int = 84):

        super(MAMLConv, self).__init__()
        self.n_way = n_way
        self.n_filters = n_filters
        # Calculate H_out after 4 ConvBlocks
        for i in range(4):
            H_in = int(np.floor(H_in/2))

        self.block1 = ConvBlock(C_in, n_filters)
        self.block2 = ConvBlock(n_filters, n_filters)
        self.block3 = ConvBlock(n_filters, n_filters)
        self.block4 = ConvBlock(n_filters, n_filters)
        self.fc = nn.Linear(n_filters*H_in*H_in, n_way)

    def forward(self, x, params_dict=None):
        if params_dict is None:
            params_dict = dict(self.named_parameters())
        for i in [1, 2, 3, 4]:
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


class BayesRegressor(nn.Module):
    """ test bayes regression
    prediction for each theta1
    model = g_z (x, theta1)
    Bayes prediction: integrate over theta1:
    y_hat = int over theta1 g_z(x, theta1)*exp(-loss_p)/Z
    loss of each contribution
    loss_p = loss(g_z(x, theta1), y)
    Partition function
    Z = sum over theta1 exp(-loss_p)
    """

    def __init__(self):
        super(BayesRegressor, self).__init__()
        self.fc1 = nn.Linear(3, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 1)
        self.rangt = []
        for i in np.arange(-15, 15, 0.5):
            for j in np.arange(-15, 15, 0.5):
                a = torch.as_tensor([i, j], dtype=torch.float32)
                self.rangt.append(a)
        self.rangt = torch.stack(self.rangt, dim=0)
        self.unnorm_ps = []
        self.lamb = 1.5

    def forward(self, x, y):
        self.unnorm_ps = []
        y_hat = torch.zeros(1)
        self.Z = torch.zeros(1)
        rangt = self.rangt + self.max_theta
        # approx integral
        for theta1 in rangt:
            stack = []
            for i in range(x.size(0)):
                stack.append(theta1)
            stack = torch.stack(stack, dim=0)
            o = torch.cat((stack, x), dim=1)
            o = F.relu(self.fc1(o))
            o = F.relu(self.fc2(o))
            o = self.fc3(o)
            unnorm_p = torch.exp(-F.mse_loss(o, y))*self.lamb
            y_hat = y_hat + o*unnorm_p
            self.Z = self.Z + unnorm_p
            self.unnorm_ps.append(unnorm_p)
        self.unnorm_ps = torch.as_tensor(self.unnorm_ps)
        y_hat = y_hat/self.Z
        return y_hat

    def predict(self, x):
        y_hat = torch.zeros(1)
        rangt = self.rangt + self.max_theta
        for theta1, unnorm_p in zip(rangt, self.unnorm_ps):
            stack = []
            for i in range(x.size(0)):
                stack.append(theta1)
            stack = torch.stack(stack, dim=0)
            o = torch.cat((stack, x), dim=1)
            o = F.relu(self.fc1(o))
            o = F.relu(self.fc2(o))
            o = self.fc3(o)
            y_hat = y_hat + o*unnorm_p
        y_hat = y_hat/self.Z
        return y_hat

    def prep_int_approx(self, x, y):
        # find best theta to approx integral
        with torch.no_grad():
            self.max_theta = torch.zeros(2)
            for j in range(10):
                # y_hat = self.forward(x, y)
                max_id = torch.argmax(self.unnorm_ps)
                self.max_theta = self.rangt[max_id]
