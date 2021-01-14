from typing import Optional

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
        decoder_hidden_size: int,
        decoder_layer_num: int,
    ):
        super().__init__()

        self.speaker_embedder = (
            nn.Embedding(
                num_embeddings=speaker_size,
                embedding_dim=speaker_embedding_size,
            )
            if self.with_speaker
            else None
        )

        input_size = input_feature_size + (
            speaker_embedding_size if self.with_speaker else 0
        )
        self.decoder = nn.GRU(
            input_size=input_size,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_layer_num,
            batch_first=True,
            bidirectional=True,
        )

        self.post = nn.Linear(
            in_features=decoder_hidden_size * 2,
            out_features=output_size,
        )

    @property
    def with_speaker(self):
        return self.speaker_size > 0

    def forward(self, f0: Tensor, phoneme: Tensor, speaker_id: Optional[Tensor]):
        feature = torch.cat((f0, phoneme), dim=2)

        if self.with_speaker:
            speaker_id = speaker_id.unsqueeze(dim=1)  # (batch_size, 1, ?)
            speaker_feature = speaker_id.expand(
                speaker_id.shape[0], feature.shape[1], speaker_id.shape[2]
            )  # (batch_size, length, ?)
            feature = torch.cat(
                (feature, speaker_feature), dim=2
            )  # (batch_size, length, ?)

        h, _ = self.decoder(feature)
        return self.post(h)


def create_predictor(config: NetworkConfig):
    return Predictor(
        input_feature_size=config.input_feature_size,
        output_size=config.output_size,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        decoder_hidden_size=config.decoder_hidden_size,
        decoder_layer_num=config.decoder_layer_num,
    )
