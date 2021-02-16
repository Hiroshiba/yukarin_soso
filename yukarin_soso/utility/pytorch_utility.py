from copy import deepcopy
from typing import Any, Callable, Dict

import torch
import torch_optimizer
from pytorch_trainer.dataset import convert
from pytorch_trainer.training.updaters.standard_updater import StandardUpdater
from torch import nn, optim
from torch.optim.optimizer import Optimizer

try:
    from torch.cuda import amp
except ImportError:
    pass


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


class AmpUpdater(StandardUpdater):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = amp.GradScaler()

    def update_core(self):
        iterator = self._iterators["main"]
        batch = iterator.next()
        in_arrays = convert._call_converter(self.converter, batch, self.device)

        optimizer = self._optimizers["main"]
        model = self._models["main"]
        loss_func = self.loss_func or model

        for model in self._models.values():
            model.train()
        optimizer.zero_grad()

        with amp.autocast():
            if isinstance(in_arrays, tuple):
                loss = loss_func(*in_arrays)
            elif isinstance(in_arrays, dict):
                loss = loss_func(**in_arrays)
            else:
                loss = loss_func(in_arrays)

        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["scaler"] = self.scaler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.scaler.load_state_dict(state_dict["scaler"])
