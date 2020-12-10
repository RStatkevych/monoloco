import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))


class Swish(nn.Module):

    def forward(self, x):
        return x * F.sigmoid(x)




class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5, activation_function=nn.ReLU):
        nn.Module.__init__(self)
        self.l_size = linear_size

        if activation_function is nn.PReLU:
            self.relu = activation_function(linear_size)
        else:
            self.relu = activation_function()
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out



class LinearShorter(Linear):

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


layers_dict = {
    'linear': Linear,
    'linear_shorter': LinearShorter
}



class LinearModel(nn.Module):
    """
    Architecture inspired by https://github.com/una-dinosauria/3d-pose-baseline
    Pytorch implementation from: https://github.com/weigq/3d_pose_baseline_pytorch
    """

    def __init__(self, input_size, output_size=2, linear_size=256, p_dropout=0.2, num_stage=3, activation_function=nn.ReLU, linear_type='linear'):
        super(LinearModel, self).__init__()
        print(f"Activation function {activation_function}, input_size = {input_size}, linear_size={linear_size}")

        self.input_size = input_size
        self.output_size = output_size
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
        
        lin_layer = layers_dict.get(linear_type, Linear)
        print(lin_layer)
        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(lin_layer(self.linear_size, self.p_dropout, activation_function))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)
        if activation_function is nn.PReLU:
            self.relu = activation_function(linear_size)
        else:
            self.relu = activation_function()
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        y = self.w2(y)
        return y
