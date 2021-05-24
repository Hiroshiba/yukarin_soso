from enum import Enum
from typing import List, Optional

import torch.nn.functional as F
from torch import Tensor, nn


class EncoderType(str, Enum):
    cnn = "cnn"
    skip_cnn = "skip_cnn"
    residual_bottleneck_cnn = "res_bot_cnn"
    uniform_skip_dilated_cnn = "uni_skip_dil_cnn"
    gru = "gru"
    uni_gru = "uni_gru"


class BaseEncoder(nn.Module):
    output_hidden_size: int


class CNN(BaseEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        layer_num: int,
    ):
        super().__init__()
        self.output_hidden_size = hidden_size

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


class SkipCNN(BaseEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        layer_num: int,
    ):
        super().__init__()
        self.output_hidden_size = hidden_size

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


class ResidualBottleneckCNN(BaseEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        layer_num: int,
    ):
        super().__init__()
        self.output_hidden_size = hidden_size

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


class UniformSkipDilatedCNN(BaseEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        layer_num: int,
    ):
        super().__init__()
        self.output_hidden_size = hidden_size

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
                        dilation=kernel_size ** i,
                        padding=kernel_size ** i * (kernel_size // 2),
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
        h = self.pre(x)
        for conv in self.conv_list:
            h = h + conv(F.silu(h))
        h = h.transpose(1, 2)
        return h


class GRU(BaseEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        layer_num: int,
    ):
        super().__init__()
        self.output_hidden_size = hidden_size * 2

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layer_num,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: Tensor):
        """
        :param x: float (batch_size, length, ?)
        """
        h, _ = self.rnn(x)
        return h


class UniGRU(BaseEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        layer_num: int,
    ):
        super().__init__()
        self.output_hidden_size = hidden_size

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layer_num,
            batch_first=True,
            bidirectional=False,
        )

    def forward(
        self, x: Tensor, hidden: Optional[Tensor] = None, return_hidden: bool = False
    ):
        """
        :param x: float (batch_size, length, ?)
        """
        output, hidden = self.rnn(x, hidden)
        if not return_hidden:
            return output
        else:
            return output, hidden


def create_encoder(
    type: EncoderType,
    input_size: int,
    hidden_size: int,
    layer_num: int,
    kernel_size: int = 0,
) -> BaseEncoder:
    kwargs = dict(
        input_size=input_size,
        hidden_size=hidden_size,
        layer_num=layer_num,
        kernel_size=kernel_size,
    )

    if type == EncoderType.gru or type == EncoderType.uni_gru:
        assert kernel_size == 0
        kwargs.pop("kernel_size")

    return {
        EncoderType.cnn: CNN,
        EncoderType.skip_cnn: SkipCNN,
        EncoderType.residual_bottleneck_cnn: ResidualBottleneckCNN,
        EncoderType.uniform_skip_dilated_cnn: UniformSkipDilatedCNN,
        EncoderType.gru: GRU,
        EncoderType.uni_gru: UniGRU,
    }[type](**kwargs)
