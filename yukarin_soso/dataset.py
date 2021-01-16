import json
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy
from acoustic_feature_extractor.data.sampling_data import SamplingData
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data._utils.collate import default_convert

from yukarin_soso.config import DatasetConfig


def resample(rate: float, data: SamplingData):
    length = int(len(data.array) / data.rate * rate)
    indexes = (numpy.random.rand() + numpy.arange(length)) * (data.rate / rate)
    return data.array[indexes.astype(int)]


@dataclass
class Input:
    f0: SamplingData
    phoneme: SamplingData
    spec: SamplingData
    silence: SamplingData


@dataclass
class LazyInput:
    f0_path: SamplingData
    phoneme_path: SamplingData
    spec_path: SamplingData
    silence_path: SamplingData

    def generate(self):
        return Input(
            f0=SamplingData.load(self.f0_path),
            phoneme=SamplingData.load(self.phoneme_path),
            spec=SamplingData.load(self.spec_path),
            silence=SamplingData.load(self.silence_path),
        )


class FeatureDataset(Dataset):
    def __init__(
        self,
        inputs: List[Union[Input, LazyInput]],
        sampling_length: int,
    ):
        self.inputs = inputs
        self.sampling_length = sampling_length

    @staticmethod
    def extract_input(
        sampling_length: int,
        f0_data: SamplingData,
        phoneme_data: SamplingData,
        spec_data: SamplingData,
        silence_data: SamplingData,
    ):
        rate = spec_data.rate

        f0 = resample(rate=rate, data=f0_data)
        phoneme = resample(rate=rate, data=phoneme_data)
        silence = resample(rate=rate, data=silence_data)

        assert numpy.abs(len(spec_data.array) - len(f0)) < 5
        assert numpy.abs(len(spec_data.array) - len(phoneme)) < 5
        assert numpy.abs(len(spec_data.array) - len(silence)) < 5

        length = min(len(spec_data.array), len(f0), len(phoneme), len(silence))

        for _ in range(10000):
            offset = numpy.random.randint(length - sampling_length + 1)
            s = numpy.squeeze(silence[offset : offset + sampling_length])
            if not s.all():
                break
        else:
            raise Exception("cannot pick not silence data")

        if silence.ndim == 2:
            silence = numpy.squeeze(silence, axis=1)

        return dict(
            f0=f0[offset : offset + sampling_length].astype(numpy.float32),
            phoneme=phoneme[offset : offset + sampling_length].astype(numpy.float32),
            spec=spec_data.array[offset : offset + sampling_length].astype(
                numpy.float32
            ),
            silence=silence[offset : offset + sampling_length],
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        input = self.inputs[i]
        if isinstance(input, LazyInput):
            input = input.generate()

        return self.extract_input(
            sampling_length=self.sampling_length,
            f0_data=input.f0,
            phoneme_data=input.phoneme,
            spec_data=input.spec,
            silence_data=input.silence,
        )


class SpeakerFeatureDataset(Dataset):
    def __init__(self, dataset: FeatureDataset, speaker_ids: List[int]):
        assert len(dataset) == len(speaker_ids)
        self.dataset = dataset
        self.speaker_ids = speaker_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        d = self.dataset[i]
        d["speaker_id"] = numpy.array(self.speaker_ids[i], dtype=numpy.int64)
        return d


class TensorWrapperDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return default_convert(self.dataset[i])


def create_dataset(config: DatasetConfig):
    f0_paths = {Path(p).stem: Path(p) for p in glob(config.f0_glob)}
    fn_list = sorted(f0_paths.keys())
    assert len(fn_list) > 0

    phoneme_paths = {Path(p).stem: Path(p) for p in glob(config.phoneme_glob)}
    assert set(fn_list) == set(phoneme_paths.keys())

    spec_paths = {Path(p).stem: Path(p) for p in glob(config.spec_glob)}
    assert set(fn_list) == set(spec_paths.keys())

    silence_paths = {Path(p).stem: Path(p) for p in glob(config.silence_glob)}
    assert set(fn_list) == set(silence_paths.keys())

    speaker_ids: Optional[Dict[str, int]] = None
    if config.speaker_dict_path is not None:
        fn_each_speaker: Dict[str, List[str]] = json.loads(
            config.speaker_dict_path.read_text()
        )
        assert config.num_speaker == len(fn_each_speaker)

        speaker_ids = {
            fn: speaker_id
            for speaker_id, (_, fns) in enumerate(fn_each_speaker.items())
            for fn in fns
        }
        assert set(fn_list).issubset(set(speaker_ids.keys()))

    numpy.random.RandomState(config.seed).shuffle(fn_list)

    test_num = config.test_num
    trains = fn_list[test_num:]
    tests = fn_list[:test_num]

    def _dataset(fns, for_test=False):
        inputs = [
            LazyInput(
                f0_path=f0_paths[fn],
                phoneme_path=phoneme_paths[fn],
                spec_path=spec_paths[fn],
                silence_path=silence_paths[fn],
            )
            for fn in fns
        ]

        dataset = FeatureDataset(
            inputs=inputs,
            sampling_length=config.sampling_length,
        )

        if speaker_ids is not None:
            dataset = SpeakerFeatureDataset(
                dataset=dataset,
                speaker_ids=[speaker_ids[fn] for fn in fns],
            )

        dataset = TensorWrapperDataset(dataset)

        if for_test:
            dataset = ConcatDataset([dataset] * config.test_trial_num)

        return dataset

    return {
        "train": _dataset(trains),
        "test": _dataset(tests, for_test=True),
    }
