import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy
from torch.utils.data._utils.collate import default_convert
from torch.utils.data.dataset import ConcatDataset, Dataset

from yukarin_soso.config import DatasetConfig


@dataclass
class InputData:
    feature: numpy.ndarray
    target: numpy.ndarray


@dataclass
class LazyInputData:
    feature_path: Path
    target_path: Path

    def generate(self):
        return InputData(
            feature=numpy.load(self.feature_path, allow_pickle=True),
            target=numpy.load(self.target_path, allow_pickle=True),
        )


class FeatureTargetDataset(Dataset):
    def __init__(
        self,
        datas: Sequence[Union[InputData, LazyInputData]],
    ):
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        data = self.datas[i]
        if isinstance(input, LazyInputData):
            data = data.generate()

        return default_convert(
            dict(
                feature=data.feature,
                target=data.target,
            )
        )


def create_dataset(config: DatasetConfig):
    feature_paths = [Path(p) for p in sorted(glob.glob(str(config.feature_glob)))]
    assert len(feature_paths) > 0

    target_paths = [Path(p) for p in sorted(glob.glob(str(config.target_glob)))]
    assert len(feature_paths) == len(target_paths)

    features = [
        LazyInputData(
            feature_path=feature_path,
            target_path=target_path,
        )
        for feature_path, target_path in zip(feature_paths, target_paths)
    ]

    if config.seed is not None:
        numpy.random.RandomState(config.seed).shuffle(features)

    tests, trains = features[: config.test_num], features[config.test_num :]

    def dataset_wrapper(datas, is_eval: bool):
        dataset = FeatureTargetDataset(
            datas=datas,
        )
        if is_eval:
            dataset = ConcatDataset([dataset] * config.eval_times_num)
        return dataset

    return {
        "train": dataset_wrapper(trains, is_eval=False),
        "test": dataset_wrapper(tests, is_eval=False),
        "eval": dataset_wrapper(tests, is_eval=True),
    }
