from typing import List, Optional

import torch
from torch import Tensor, nn
from yukarin_soso.config import NetworkConfig


class Predictor(nn.Module):
    def __init__(
        self,
        input_feature_size: int,
        output_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
        cnn_hidden_size_list: List[int],
        cnn_kernel_size_list: List[int],
        rnn_hidden_size: int,
        rnn_layer_num: int,
    ):
        cnn_layer_num = len(cnn_hidden_size_list)
        assert len(cnn_kernel_size_list) == cnn_layer_num

        super().__init__()

        self.with_speaker = speaker_size > 0

        self.speaker_embedder = (
            nn.Embedding(
                num_embeddings=speaker_size,
                embedding_dim=speaker_embedding_size,
            )
            if self.with_speaker
            else None
        )

        # cnn
        input_size = input_feature_size + (
            speaker_embedding_size if self.with_speaker else 0
        )

        cnn: List[nn.Module] = []
        for i in range(cnn_layer_num):
            cnn.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        in_channels=(
                            cnn_hidden_size_list[i - 1] if i > 0 else input_size
                        ),
                        out_channels=cnn_hidden_size_list[i],
                        kernel_size=cnn_kernel_size_list[i],
                        padding=cnn_kernel_size_list[i] // 2,
                    )
                )
            )
            if i < cnn_layer_num - 1:
                cnn.append(nn.SiLU(inplace=True))
        self.cnn = nn.Sequential(*cnn)

        # rnn
        self.rnn = nn.GRU(
            input_size=cnn_hidden_size_list[-1],
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layer_num,
            batch_first=True,
            bidirectional=True,
        )

        # post
        self.post = nn.Linear(
            in_features=rnn_hidden_size * 2,
            out_features=output_size,
        )

    def forward(self, f0: Tensor, phoneme: Tensor, speaker_id: Optional[Tensor]):
        feature = torch.cat((f0, phoneme), dim=2)  # (batch_size, length, ?)

        if self.with_speaker:
            speaker_id = self.speaker_embedder(speaker_id)
            speaker_id = speaker_id.unsqueeze(dim=1)  # (batch_size, 1, ?)
            speaker_feature = speaker_id.expand(
                speaker_id.shape[0], feature.shape[1], speaker_id.shape[2]
            )  # (batch_size, length, ?)
            feature = torch.cat(
                (feature, speaker_feature), dim=2
            )  # (batch_size, length, ?)

        h = self.cnn(feature.transpose(1, 2)).transpose(1, 2)  # (batch_size, length, ?)
        h, _ = self.rnn(h)
        return self.post(h)


def create_predictor(config: NetworkConfig):
    return Predictor(
        input_feature_size=config.input_feature_size,
        output_size=config.output_size,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        cnn_hidden_size_list=config.cnn_hidden_size_list,
        cnn_kernel_size_list=config.cnn_kernel_size_list,
        rnn_hidden_size=config.rnn_hidden_size,
        rnn_layer_num=config.rnn_layer_num,
    )
