from pathlib import Path
from typing import List, Optional

import numpy
import pytest
from acoustic_feature_extractor.data.phoneme import JvsPhoneme
from acoustic_feature_extractor.data.sampling_data import SamplingData
from yukarin_soso.dataset import F0ProcessMode, FeatureDataset, f0_mean

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


@pytest.fixture()
def volume_path():
    return get_data_directory() / "volume_001.npy"


@pytest.mark.parametrize(
    "f0,rate,split_second_list,weight,expected",
    [
        (
            numpy.arange(1, 11, dtype=numpy.float32),
            2,
            [1, 2, 3, 4],
            None,
            numpy.repeat(numpy.arange(1.5, 10, step=2, dtype=numpy.float32), 2),
        ),
        (
            numpy.array([0, 0, 0, 1, 1, 1], dtype=numpy.float32),
            2,
            [1, 2],
            None,
            numpy.array([0, 0, 1, 1, 1, 1], dtype=numpy.float32),
        ),
        (
            numpy.arange(1, 11, dtype=numpy.float32),
            1.5,
            [1, 2, 3, 4],
            None,
            numpy.array(
                [1, 2.5, 2.5, 4, 5.5, 5.5, 8.5, 8.5, 8.5, 8.5], dtype=numpy.float32
            ),
        ),
        (
            numpy.array([1, 2, 1, 2, 1, 2], dtype=numpy.float32),
            2,
            [1, 2],
            numpy.array([0, 1, 1, 1, 1, 0], dtype=numpy.float32),
            numpy.array([2, 2, 1.5, 1.5, 1, 1], dtype=numpy.float32),
        ),
    ],
)
def test_f0_mean(
    f0: numpy.ndarray,
    rate: float,
    split_second_list: List[float],
    expected: numpy.ndarray,
    weight: numpy.ndarray,
):
    output = f0_mean(
        rate=rate,
        f0=f0,
        split_second_list=split_second_list,
        weight=weight,
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

    f0 = numpy.arange(int(second * f0_rate)).reshape(-1, 1).astype(numpy.float32)
    f0_data = SamplingData(array=f0, rate=f0_rate)

    phoneme = (
        numpy.arange(int(second * phoneme_rate)).reshape(-1, 1).astype(numpy.float32)
    )
    phoneme_data = SamplingData(array=phoneme, rate=phoneme_rate)

    spec = numpy.arange(int(second * spec_rate)).reshape(-1, 1).astype(numpy.float32)
    spec_data = SamplingData(array=spec, rate=spec_rate)

    silence = numpy.zeros(int(second * silence_rate)).astype(bool)
    silence_data = SamplingData(array=silence, rate=silence_rate)

    phoneme_list_data = None
    volume_data = None
    f0_process_mode = F0ProcessMode.normal
    time_mask_max_second = 0
    time_mask_num = 0

    FeatureDataset.extract_input(
        sampling_length=sampling_length,
        f0_data=f0_data,
        phoneme_data=phoneme_data,
        spec_data=spec_data,
        silence_data=silence_data,
        phoneme_list_data=phoneme_list_data,
        volume_data=volume_data,
        f0_process_mode=f0_process_mode,
        time_mask_max_second=time_mask_max_second,
        time_mask_num=time_mask_num,
    )


@pytest.mark.parametrize(
    "sampling_length,f0_process_mode,time_mask_max_second,time_mask_num",
    [
        (256, F0ProcessMode.normal, 0, 0),
        (256, F0ProcessMode.phoneme_mean, 0, 0),
        (256, F0ProcessMode.mora_mean, 0, 0),
        (256, F0ProcessMode.normal, 0.5, 1),
        (256, F0ProcessMode.voiced_mora_mean, 0, 0),
    ],
)
def test_extract_input_with_dataset(
    sampling_length: int,
    f0_path: Path,
    phoneme_path: Path,
    phoneme_list_path: Path,
    silence_path: Path,
    spectrogram_path: Path,
    volume_path: Path,
    f0_process_mode: F0ProcessMode,
    time_mask_max_second: float,
    time_mask_num: int,
):
    f0 = SamplingData.load(f0_path)
    phoneme = SamplingData.load(phoneme_path)
    phoneme_list = JvsPhoneme.load_julius_list(phoneme_list_path)
    silence = SamplingData.load(silence_path)
    spectrogram = SamplingData.load(spectrogram_path)
    volume_data = SamplingData.load(volume_path)

    FeatureDataset.extract_input(
        sampling_length=sampling_length,
        f0_data=f0,
        phoneme_data=phoneme,
        spec_data=spectrogram,
        silence_data=silence,
        phoneme_list_data=phoneme_list,
        volume_data=volume_data,
        f0_process_mode=f0_process_mode,
        time_mask_max_second=time_mask_max_second,
        time_mask_num=time_mask_num,
    )
