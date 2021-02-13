import argparse
import importlib
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict

import optuna
import yaml
from optuna.pruners import PercentilePruner
from optuna.storages import RDBStorage

from yukarin_soso.config import Config
from yukarin_soso.trainer import create_trainer
from yukarin_soso.utility.trainer_utility import PruningExtension


def param_dict_to_name(param_dict: Dict[str, Any]):
    return ",".join(
        f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
        for k, v in sorted(param_dict.items())
    )


def modify_config(config: Config, optuna_config_path: Path, trial: optuna.Trial):
    sys.path.append(str(optuna_config_path.parent))
    config = importlib.import_module(optuna_config_path.stem).modify_config(
        config=config, trial=trial
    )
    return config


def objective(
    trial: optuna.Trial,
    name: str,
    config_yaml_path: Path,
    optuna_config_path: Path,
    root_output: Path,
):
    with config_yaml_path.open() as f:
        config = Config.from_dict(yaml.safe_load(f))

    config = modify_config(
        config=config, optuna_config_path=optuna_config_path, trial=trial
    )
    postfix = param_dict_to_name(trial.params)
    config.project.name = f"{name}-" + postfix
    output = root_output.joinpath(f"{trial.number}-" + config.project.name)

    trainer = create_trainer(config=config, output=output)
    trainer.extend(
        PruningExtension(
            trial=trial,
            observation_key=config.train.optuna["key"],
            pruner_trigger=(config.train.optuna["iteration"], "iteration"),
        ),
    )
    trainer.run()

    log_last = trainer.get_extension("LogReport").log[-1]
    return log_last[config.train.optuna["key"]]


def train_optuna(
    config_yaml_path: Path,
    optuna_config_path: Path,
    root_output: Path,
    name: str,
    storage: str,
    num_trials: int,
):
    study = optuna.create_study(
        storage=RDBStorage(storage),
        pruner=PercentilePruner(25),
        study_name=name,
        load_if_exists=True,
    )
    objective_wrapper = partial(
        objective,
        name=name,
        config_yaml_path=config_yaml_path,
        optuna_config_path=optuna_config_path,
        root_output=root_output,
    )
    study.optimize(func=objective_wrapper, n_trials=num_trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml_path", type=Path)
    parser.add_argument("optuna_config_path", type=Path)
    parser.add_argument("root_output", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--storage")
    parser.add_argument("--num_trials", type=int)
    train_optuna(**vars(parser.parse_args()))
