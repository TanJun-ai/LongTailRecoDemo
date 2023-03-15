
import torch
import torch.nn as nn


def activation_layer(act_name):
    """Construct activation layers
    Args:
        act_name: str or nn.Module, name of activation function
    Returns:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'dice':
            act_layer = Dice()
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
        elif act_name.lower() == "softmax":
            act_layer = nn.Softmax(dim=1)
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError
    return act_layer

class Dice(nn.Module):
    """The Dice activation function mentioned in the `DIN paper
    https://arxiv.org/abs/1706.06978`
    """
    def __init__(self, epsilon=1e-3):
        super(Dice, self).__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor):
        # x: N * num_neurons
        avg = x.mean(dim=1)  # N
        avg = avg.unsqueeze(dim=1)  # N * 1
        var = torch.pow(x - avg, 2) + self.epsilon  # N * num_neurons
        var = var.sum(dim=1).unsqueeze(dim=1)  # N * 1
        ps = (x - avg) / torch.sqrt(var)  # N * 1
        ps = nn.Sigmoid()(ps)  # N * 1
        return ps * x + (1 - ps) * self.alpha * x

class MLP(nn.Module):
    """Multi Layer Perceptron Module, it is the most widely used module for
    learning feature. Note we default add `BatchNorm1d` and `Activation`
    `Dropout` for each `Linear` Module.
    Args:
        input dim (int): input size of the first Linear Layer.
        output_layer (bool): whether this MLP module is the output layer. If `True`, then append one Linear(*,1) module.
        dims (list): output size of Linear Layer (default=[]).
        dropout (float): probability of an element to be zeroed (default = 0.5).
        activation (str): the activation function, support `[sigmoid, relu, prelu, dice, softmax]` (default='relu').
    Shape:
        - Input: `(batch_size, input_dim)`
        - Output: `(batch_size, 1)` or `(batch_size, dims[-1])`
    """
    def __init__(self, input_dim, output_layer=True, dims=None, dropout=0, activation="relu"):
        super().__init__()
        if dims is None:
            dims = []
        layers = list()
        for i_dim in dims:
            layers.append(nn.Linear(input_dim, i_dim))
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(activation_layer(activation))
            layers.append(nn.Dropout(p=dropout))
            input_dim = i_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
