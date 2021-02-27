from typing import List, Optional

import torch
from torch import Tensor, nn
from yukarin_soso.config import CNNType, NetworkConfig
from yukarin_soso.network.encoder import CNN, ResidualBottleneckCNN, SkipCNN


class Predictor(nn.Module):
    def __init__(
        self,
        input_feature_size: int,
        output_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
        cnn_type: CNNType,
        cnn_hidden_size: int,
        cnn_kernel_size: int,
        cnn_layer_num: int,
        rnn_hidden_size: int,
        rnn_layer_num: int,
        ar_hidden_size: int,
    ):
        super().__init__()
        self.output_size = output_size

        self.speaker_embedder = (
            nn.Embedding(
                num_embeddings=speaker_size,
                embedding_dim=speaker_embedding_size,
            )
            if speaker_size > 0
            else None
        )

        input_size = input_feature_size + speaker_embedding_size

        # cnn
        self.cnn = {
            CNNType.cnn: CNN,
            CNNType.skip_cnn: SkipCNN,
            CNNType.residual_bottleneck_cnn: ResidualBottleneckCNN,
        }[cnn_type](
            input_size=input_size,
            hidden_size=cnn_hidden_size,
            kernel_size=cnn_kernel_size,
            layer_num=cnn_layer_num,
        )

        # rnn
        self.rnn = nn.GRU(
            input_size=cnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layer_num,
            batch_first=True,
            bidirectional=True,
        )

        # AR rnn
        if ar_hidden_size == 0:
            self.ar_rnn = None
        else:
            self.ar_rnn = nn.GRU(
                input_size=rnn_hidden_size * 2 + output_size,
                hidden_size=ar_hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )

        # post
        input_size = rnn_hidden_size * 2 if ar_hidden_size == 0 else ar_hidden_size
        self.post = nn.Linear(
            in_features=input_size,
            out_features=output_size,
        )

    def forward_encoder(
        self,
        f0: Tensor,
        phoneme: Tensor,
        speaker_id: Optional[Tensor],
    ):
        feature = torch.cat((f0, phoneme), dim=2)  # (batch_size, length, ?)

        if self.speaker_embedder is not None:
            speaker_id = self.speaker_embedder(speaker_id)
            speaker_id = speaker_id.unsqueeze(dim=1)  # (batch_size, 1, ?)
            speaker_feature = speaker_id.expand(
                speaker_id.shape[0], feature.shape[1], speaker_id.shape[2]
            )  # (batch_size, length, ?)
            feature = torch.cat(
                (feature, speaker_feature), dim=2
            )  # (batch_size, length, ?)

        h = self.cnn(feature)  # (batch_size, length, ?)
        h, _ = self.rnn(h)  # (batch_size, length, ?)
        return h

    def forward(
        self,
        f0: Tensor,
        phoneme: Tensor,
        spec: Tensor,
        speaker_id: Optional[Tensor],
    ):
        h = self.forward_encoder(
            f0=f0,
            phoneme=phoneme,
            speaker_id=speaker_id,
        )  # (batch_size, length, ?)

        if self.ar_rnn is not None:
            h = torch.cat((h, spec), dim=2)  # (batch_size, length, ?)
            h, _ = self.ar_rnn(h)  # (batch_size, length, ?)

        return self.post(h)

    def inference(
        self,
        f0: Tensor,
        phoneme: Tensor,
        speaker_id: Optional[Tensor],
    ):
        batch_size = len(phoneme)

        h = self.forward_encoder(
            f0=f0,
            phoneme=phoneme,
            speaker_id=speaker_id,
        )

        if self.ar_rnn is not None:
            spec_list: List[Tensor] = []

            spec = torch.zeros(batch_size, 1, self.output_size, dtype=h.dtype).to(
                h.device
            )  # (batch, 1, ?)
            hidden = None
            for i in range(h.shape[1]):
                h_one = h[:, i : i + 1, :]  # (batch, 1, ?)
                h_one = torch.cat((h_one, spec), dim=2)  # (batch, 1, ?)
                h_one, hidden = self.ar_rnn(h_one, hidden)  # (batch, 1, ?)
                spec = self.post(h_one)  # (batch, 1, ?)

                spec_list.append(spec)

            h = torch.cat(spec_list, dim=1)  # (batch, length, ?)

        else:
            h = self.post(h)  # (batch, length, ?)

        return h


def create_predictor(config: NetworkConfig):
    return Predictor(
        input_feature_size=config.input_feature_size,
        output_size=config.output_size,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        cnn_type=CNNType(config.cnn_type),
        cnn_hidden_size=config.cnn_hidden_size,
        cnn_kernel_size=config.cnn_kernel_size,
        cnn_layer_num=config.cnn_layer_num,
        rnn_hidden_size=config.rnn_hidden_size,
        rnn_layer_num=config.rnn_layer_num,
        ar_hidden_size=config.ar_hidden_size,
    )
