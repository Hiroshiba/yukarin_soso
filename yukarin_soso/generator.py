from pathlib import Path
from typing import Any, Union

import numpy
import torch
from torch import Tensor

from yukarin_soso.config import Config
from yukarin_soso.network.predictor import Predictor, create_predictor


def to_tensor(array: Union[Tensor, numpy.ndarray, Any]):
    if not isinstance(array, (Tensor, numpy.ndarray)):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        return torch.from_numpy(array)
    else:
        return array


class Generator(object):
    def __init__(
        self,
        config: Config,
        predictor: Union[Predictor, Path],
        use_gpu: bool,
    ):
        self.config = config
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor, map_location=self.device)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

    def generate(
        self,
        f0: Union[numpy.ndarray, torch.Tensor],
        phoneme: Union[numpy.ndarray, torch.Tensor],
        speaker_id: Union[numpy.ndarray, torch.Tensor] = None,
    ):
        f0 = to_tensor(f0)
        phoneme = to_tensor(phoneme)
        if speaker_id is not None:
            speaker_id = to_tensor(speaker_id)

        with torch.no_grad():
            output = self.predictor(f0=f0, phoneme=phoneme, speaker_id=speaker_id)
        return output.cpu().numpy()
