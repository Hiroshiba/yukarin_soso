from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from yukarin_soso.utility import dataclass_utility
from yukarin_soso.utility.git_utility import get_branch_name, get_commit_id


@dataclass
class DatasetConfig:
    sampling_length: int
    f0_glob: str
    phoneme_glob: str
    spec_glob: str
    silence_glob: str
    phoneme_list_glob: Optional[str]
    volume_glob: Optional[str]
    f0_process_mode: str
    time_mask_max_second: float
    time_mask_num: int
    speaker_dict_path: Optional[Path]
    num_speaker: Optional[int]
    weighted_speaker_id: Optional[int]
    speaker_weight: Optional[int]
    test_num: int
    test_trial_num: int = 1
    valid_f0_glob: Optional[str] = None
    valid_phoneme_glob: Optional[str] = None
    valid_spec_glob: Optional[str] = None
    valid_silence_glob: Optional[str] = None
    valid_phoneme_list_glob: Optional[str] = None
    valid_volume_glob: Optional[str] = None
    valid_speaker_dict_path: Optional[Path] = None
    valid_trial_num: Optional[int] = None
    valid_num: Optional[int] = None
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
    ar_hidden_size: int
    ar_layer_num: int


@dataclass
class ModelConfig:
    eliminate_silence: bool


@dataclass
class TrainConfig:
    batch_size: int
    log_iteration: int
    eval_iteration: int
    snapshot_iteration: int
    stop_iteration: int
    optimizer: Dict[str, Any]
    weight_initializer: Optional[str] = None
    step_shift: Optional[Dict[str, Any]] = None
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
    if "ar_hidden_size" not in d["network"]:
        d["network"]["ar_hidden_size"] = 0

    if "phoneme_list_glob" not in d["dataset"]:
        d["dataset"]["phoneme_list_glob"] = None

    if "f0_process_mode" not in d["dataset"]:
        d["dataset"]["f0_process_mode"] = "normal"

    if "time_mask_max_second" not in d["dataset"]:
        d["dataset"]["time_mask_max_second"] = 0

    if "time_mask_num" not in d["dataset"]:
        d["dataset"]["time_mask_num"] = 0

    if "eval_iteration" not in d["train"]:
        d["train"]["eval_iteration"] = 0

    if "volume_glob" not in d["dataset"]:
        d["dataset"]["volume_glob"] = None

    if "weighted_speaker_id" not in d["dataset"]:
        d["dataset"]["weighted_speaker_id"] = None
    if "speaker_weight" not in d["dataset"]:
        d["dataset"]["speaker_weight"] = None

    if "ar_layer_num" not in d["network"]:
        d["network"]["ar_layer_num"] = 0 if d["network"]["ar_hidden_size"] == 0 else 1
