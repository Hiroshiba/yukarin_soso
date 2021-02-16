from typing import List

import torch.nn.functional as F
from torch import Tensor, nn


class CNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        layer_num: int,
    ):
        super().__init__()

        self.pre = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=1
        )

        cnn: List[nn.Module] = []
        for i in range(layer_num):
            cnn.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        in_channels=hidden_size,
                        out_channels=hidden_size,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    )
                )
            )
            if i < layer_num - 1:
                cnn.append(nn.ReLU(inplace=True))
        self.cnn = nn.Sequential(*cnn)

    def forward(self, x: Tensor):
        """
        :param x: float (batch_size, length, ?)
        """
        return self.cnn(self.pre(x.transpose(1, 2))).transpose(1, 2)


class SkipCNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        layer_num: int,
    ):
        super().__init__()

        self.pre = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=1
        )

        self.conv_list = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv1d(
                        in_channels=hidden_size,
                        out_channels=hidden_size,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    )
                )
                for i in range(layer_num)
            ]
        )

    def forward(self, x: Tensor):
        """
        :param x: float (batch_size, length, ?)
        """
        h = x.transpose(1, 2)
        h = self.pre(h)
        for conv in self.conv_list:
            h = h + conv(F.relu(h))
        h = h.transpose(1, 2)
        return h


class ResidualBottleneckCNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        layer_num: int,
    ):
        super().__init__()

        self.pre = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=1
        )

        self.conv1_list = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv1d(
                        in_channels=hidden_size,
                        out_channels=hidden_size // 4,
                        kernel_size=1,
                    )
                )
                for _ in range(layer_num)
            ]
        )
        self.conv2_list = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv1d(
                        in_channels=hidden_size // 4,
                        out_channels=hidden_size // 4,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    )
                )
                for i in range(layer_num)
            ]
        )
        self.conv3_list = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv1d(
                        in_channels=hidden_size // 4,
                        out_channels=hidden_size,
                        kernel_size=1,
                    )
                )
                for _ in range(layer_num)
            ]
        )

    def forward(self, x: Tensor):
        """
        :param x: float (batch_size, length, ?)
        """
        h = x.transpose(1, 2)
        h = self.pre(h)
        for conv1, conv2, conv3 in zip(
            self.conv1_list, self.conv2_list, self.conv3_list
        ):
            h = h + conv3(F.relu(conv2(F.relu(conv1(F.relu(h))))))
        h = h.transpose(1, 2)
        return h
