import numpy
import pytest
from acoustic_feature_extractor.data.sampling_data import SamplingData
from yukarin_soso.dataset import FeatureDataset, resample


@pytest.mark.parametrize(
    "length,source_rate,target_rate",
    [(1000, 100, 24000 / 512), (1000, 200, 24000 / 512), (1000, 24000, 24000 / 512)],
)
def test_resample(length: int, source_rate: float, target_rate: float):
    for _ in range(10):
        array = numpy.arange(length, dtype=numpy.float32)

        output = resample(
            rate=target_rate, data=SamplingData(array=array, rate=source_rate)
        )
        expect = numpy.interp(
            (
                numpy.arange(length * target_rate // source_rate, dtype=numpy.float32)
                * source_rate
                / target_rate
            ),
            array,
            array,
        )

        assert numpy.all(
            numpy.abs(expect - output) < numpy.ceil(source_rate / target_rate)
        )


def test_extract_input():
    sampling_length = 10
    wave_length = 256 * sampling_length
    wave_rate = 24000
    second = wave_length / wave_rate

    f0_rate = 200
    phoneme_rate = 100
    spec_rate = wave_rate / 256
    silence_rate = 24000

    f0 = numpy.arange(int(second * f0_rate)).astype(numpy.float32)
    f0_data = SamplingData(array=f0, rate=f0_rate)

    phoneme = numpy.arange(int(second * phoneme_rate)).astype(numpy.float32)
    phoneme_data = SamplingData(array=phoneme, rate=phoneme_rate)

    spec = numpy.arange(int(second * spec_rate)).astype(numpy.float32)
    spec_data = SamplingData(array=spec, rate=spec_rate)

    silence = numpy.zeros(int(second * silence_rate)).astype(bool)
    silence_data = SamplingData(array=silence, rate=silence_rate)

    output = FeatureDataset.extract_input(
        sampling_length=sampling_length,
        f0_data=f0_data,
        phoneme_data=phoneme_data,
        spec_data=spec_data,
        silence_data=silence_data,
    )
