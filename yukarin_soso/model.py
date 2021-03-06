from typing import Optional

import torch
import torch.nn.functional as F
from pytorch_trainer import report
from torch import Tensor, nn

from yukarin_soso.config import ModelConfig
from yukarin_soso.network.predictor import Predictor


class Model(nn.Module):
    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(
        self,
        f0: Tensor,
        phoneme: Tensor,
        spec: Tensor,
        silence: Tensor,
        padded: Tensor,
        speaker_id: Optional[Tensor] = None,
    ):
        batch_size = spec.shape[0]

        output = self.predictor(
            f0=f0,
            phoneme=phoneme,
            spec=spec.roll(1, dims=1),
            speaker_id=speaker_id,
        )

        loss = F.l1_loss(input=output, target=spec, reduction="none")

        mask = padded
        if self.model_config.eliminate_silence:
            mask = torch.logical_or(mask, silence)
        loss = loss[~mask]

        loss = loss.mean()

        # report
        losses = dict(loss=loss)
        if not self.training:
            weight = (~mask).to(torch.float32).mean() * batch_size
            losses = {key: (l, weight) for key, l in losses.items()}
        report(losses, self)

        return loss
