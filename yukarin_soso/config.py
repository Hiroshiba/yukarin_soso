from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from yukarin_soso.utility import dataclass_utility
from yukarin_soso.utility.git_utility import get_branch_name, get_commit_id


class CNNType(str, Enum):
    cnn = "cnn"
    skip_cnn = "skip_cnn"
    residual_bottleneck_cnn = "res_bot_cnn"


@dataclass
class DatasetConfig:
    sampling_length: int
    f0_glob: str
    phoneme_glob: str
    spec_glob: str
    silence_glob: str
    speaker_dict_path: Optional[Path]
    num_speaker: Optional[int]
    test_num: int
    test_trial_num: int = 1
    seed: int = 0


@dataclass
class NetworkConfig:
    input_feature_size: int
    output_size: int
    speaker_size: int
    speaker_embedding_size: int
    cnn_type: str
    cnn_hidden_size: int
    cnn_kernel_size: int
    cnn_layer_num: int
    rnn_hidden_size: int
    rnn_layer_num: int


@dataclass
class ModelConfig:
    eliminate_silence: bool


@dataclass
class TrainConfig:
    batch_size: int
    log_iteration: int
    snapshot_iteration: int
    stop_iteration: int
    optimizer: Dict[str, Any]
    weight_initializer: Optional[str] = None
    num_processes: Optional[int] = None
    use_amp: bool = False
    use_multithread: bool = False
    optuna: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectConfig:
    name: str
    tags: Dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None


@dataclass
class Config:
    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        backward_compatible(d)
        return dataclass_utility.convert_from_dict(cls, d)

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_utility.convert_to_dict(self)

    def add_git_info(self):
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: Dict[str, Any]):
    pass
