from pathlib import Path
from typing import Union

import numpy
import torch

from yukarin_soso.config import Config
from yukarin_soso.network.predictor import Predictor, create_predictor


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
            state_dict = torch.load(predictor)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

    def generate(
        self,
        feature: Union[numpy.ndarray, torch.Tensor],
    ):
        if isinstance(feature, numpy.ndarray):
            feature = torch.from_numpy(feature)
        feature = feature.to(self.device)

        with torch.no_grad():
            output = self.predictor(feature.unsqueeze(0))[0]
        return output.numpy()
