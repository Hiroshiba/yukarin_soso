from typing import Optional

import numpy
from pytorch_trainer import report
from torch import Tensor, nn

from yukarin_soso.generator import Generator

_logdb_const = 10.0 / numpy.log(10.0) * numpy.sqrt(2.0)


def _mcd(x: numpy.ndarray, y: numpy.ndarray) -> float:
    z = x - y
    r = numpy.sqrt((z * z).sum(axis=1)).mean()
    return _logdb_const * r


class GenerateEvaluator(nn.Module):
    def __init__(
        self,
        generator: Generator,
    ):
        super().__init__()
        self.generator = generator

    def __call__(
        self,
        f0: Tensor,
        phoneme: Tensor,
        spec: Tensor,
        silence: Tensor,
        padded: Tensor,
        speaker_id: Optional[Tensor] = None,
    ):
        batch_size = len(spec)
        numpy_padded = padded.cpu().numpy()

        out_spec = self.generator.generate(
            f0=f0,
            phoneme=phoneme,
            speaker_id=speaker_id,
        )
        out_spec = out_spec[~numpy_padded]

        in_spec = spec.cpu().numpy()[~numpy_padded]

        diff = numpy.abs(out_spec - in_spec).mean()
        mcd = _mcd(out_spec, in_spec)

        weight = (~numpy_padded).mean() * batch_size
        scores = {"diff": (diff, weight), "mcd": (mcd, weight)}

        report(scores, self)
        return scores
