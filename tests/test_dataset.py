from pathlib import Path
from typing import List

import numpy
import pytest
from acoustic_feature_extractor.data.phoneme import JvsPhoneme
from acoustic_feature_extractor.data.sampling_data import SamplingData
from yukarin_soso.dataset import F0ProcessMode, FeatureDataset, f0_mean, resample

from tests.utility import get_data_directory


@pytest.fixture()
def f0_path():
    return get_data_directory() / "f0_001.npy"


@pytest.fixture()
def phoneme_path():
    return get_data_directory() / "phoneme_001.npy"


@pytest.fixture()
def phoneme_list_path():
    return get_data_directory() / "phoneme_list_001.lab"


@pytest.fixture()
def silence_path():
    return get_data_directory() / "silence_001.npy"


@pytest.fixture()
def spectrogram_path():
    return get_data_directory() / "spectrogram_001.npy"


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


@pytest.mark.parametrize(
    "f0,rate,split_second_list,expected",
    [
        (
            numpy.arange(1, 11, dtype=numpy.float32),
            2,
            [1, 2, 3, 4],
            numpy.repeat(numpy.arange(1.5, 10, step=2, dtype=numpy.float32), 2),
        ),
        (
            numpy.array([0, 0, 0, 1, 1, 1], dtype=numpy.float32),
            2,
            [1, 2],
            numpy.array([0, 0, 1, 1, 1, 1], dtype=numpy.float32),
        ),
        (
            numpy.arange(1, 11, dtype=numpy.float32),
            1.5,
            [1, 2, 3, 4],
            numpy.array(
                [1, 2.5, 2.5, 4, 5.5, 5.5, 8.5, 8.5, 8.5, 8.5], dtype=numpy.float32
            ),
        ),
    ],
)
def test_f0_mean(
    f0: numpy.ndarray,
    rate: float,
    split_second_list: List[float],
    expected: numpy.ndarray,
):
    output = f0_mean(
        rate=rate,
        f0=f0,
        split_second_list=split_second_list,
    )

    numpy.testing.assert_allclose(output, expected)


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


@pytest.mark.parametrize(
    "sampling_length,f0_process_mode",
    [(256, F0ProcessMode.normal), (256, F0ProcessMode.phoneme_mean)],
)
def test_extract_input_with_dataset(
    sampling_length: int,
    f0_path: Path,
    phoneme_path: Path,
    phoneme_list_path: Path,
    silence_path: Path,
    spectrogram_path: Path,
    f0_process_mode: F0ProcessMode,
):
    f0 = SamplingData.load(f0_path)
    phoneme = SamplingData.load(phoneme_path)
    phoneme_list = JvsPhoneme.load_julius_list(phoneme_list_path)
    silence = SamplingData.load(silence_path)
    spectrogram = SamplingData.load(spectrogram_path)

    sampling_length = 256

    FeatureDataset.extract_input(
        sampling_length=sampling_length,
        f0_data=f0,
        phoneme_data=phoneme,
        spec_data=spectrogram,
        silence_data=silence,
        phoneme_list_data=phoneme_list,
        f0_process_mode=f0_process_mode,
    )
