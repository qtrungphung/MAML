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
