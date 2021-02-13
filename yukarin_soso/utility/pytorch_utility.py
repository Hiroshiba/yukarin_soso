from copy import deepcopy
from typing import Any, Callable, Dict

import torch
import torch_optimizer
from torch import nn, optim
from torch.optim.optimizer import Optimizer


def init_weights(model: torch.nn.Module, name: str):
    def _init_weights(layer: nn.Module):
        initializer: Callable
        if name == "uniform":
            initializer = torch.nn.init.uniform_
        elif name == "normal":
            initializer = torch.nn.init.normal_
        elif name == "xavier_uniform":
            initializer = torch.nn.init.xavier_uniform_
        elif name == "xavier_normal":
            initializer = torch.nn.init.xavier_normal_
        elif name == "kaiming_uniform":
            initializer = torch.nn.init.kaiming_uniform_
        elif name == "kaiming_normal":
            initializer = torch.nn.init.kaiming_normal_
        elif name == "orthogonal":
            initializer = torch.nn.init.orthogonal_
        elif name == "sparse":
            initializer = torch.nn.init.sparse_
        else:
            raise ValueError(name)

        for key, param in layer.named_parameters():
            if "weight" in key:
                initializer(param)

    model.apply(_init_weights)


def make_optimizer(config_dict: Dict[str, Any], model: nn.Module):
    cp: Dict[str, Any] = deepcopy(config_dict)
    n = cp.pop("name").lower()

    optimizer: Optimizer
    if n == "adam":
        optimizer = optim.Adam(model.parameters(), **cp)
    elif n == "radam":
        optimizer = torch_optimizer.RAdam(model.parameters(), **cp)
    elif n == "ranger":
        optimizer = torch_optimizer.Ranger(model.parameters(), **cp)
    elif n == "sgd":
        optimizer = optim.SGD(model.parameters(), **cp)
    else:
        raise ValueError(n)

    return optimizer
