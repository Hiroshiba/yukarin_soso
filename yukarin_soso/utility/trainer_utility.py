from typing import Tuple, Union

import optuna
import pytorch_trainer
import torch
from pytorch_trainer import reporter
from pytorch_trainer.iterators import (
    MultiprocessIterator,
    MultithreadIterator,
    SerialIterator,
)
from pytorch_trainer.training.extension import Extension
from pytorch_trainer.training.triggers import IntervalTrigger, ManualScheduleTrigger
from pytorch_trainer.training.util import get_trigger
from torch.utils.data import Dataset


def create_iterator(
    dataset: Dataset,
    batch_size: int,
    for_train: bool = True,
    for_eval: bool = False,
    eval_batch_size: int = None,
    num_processes: int = None,
    use_multithread: bool = False,
):
    if not for_eval or eval_batch_size is None:
        batch_size = batch_size
    else:
        batch_size = eval_batch_size

    if num_processes == 0:
        return SerialIterator(
            dataset,
            batch_size,
            repeat=for_train,
            shuffle=for_train,
        )
    else:
        if not use_multithread:
            return MultiprocessIterator(
                dataset,
                batch_size,
                repeat=for_train,
                shuffle=for_train,
                n_processes=num_processes,
                dataset_timeout=60 * 15,
            )
        else:
            return MultithreadIterator(
                dataset,
                batch_size,
                repeat=for_train,
                shuffle=for_train,
                n_threads=num_processes,
            )


class BetterValueTrigger(object):
    def __init__(self, key, compare, stock_num=5, trigger=(1, "epoch")):
        self._key = key
        self._better_values = []
        self._interval_trigger = get_trigger(trigger)
        self._init_summary()
        self._compare = compare
        self._stock_num = stock_num

    def __call__(self, trainer):
        observation = trainer.observation
        summary = self._summary
        key = self._key
        if key in observation:
            summary.add({key: observation[key]})

        if not self._interval_trigger(trainer):
            return False

        stats = summary.compute_mean()
        value = float(stats[key])  # copy to CPU
        self._init_summary()

        flag = False
        for i in range(len(self._better_values)):
            if self._compare(self._better_values[i], value):
                self._better_values.insert(i, value)
                flag = True
                break

        if not flag and len(self._better_values) < self._stock_num:
            self._better_values.append(value)
            flag = True

        if len(self._better_values) > self._stock_num:
            self._better_values = self._better_values[: self._stock_num]

        return flag

    def _init_summary(self):
        self._summary = reporter.DictSummary()

    def state_dict(self):
        return {
            "interval_trigger": self._interval_trigger.state_dict(),
            "summary": self._summary.state_dict(),
            "better_values": self._better_values,
            "stock_num": self._stock_num,
        }

    def load_state_dict(self, state_dict):
        self._interval_trigger.load_state_dict(state_dict["interval_trigger"])
        self._summary.load_state_dict(state_dict["summary"])
        self._better_values = state_dict["better_values"]
        self._stock_num = state_dict["stock_num"]


class HighValueTrigger(BetterValueTrigger):
    def __init__(self, key, stock_num=5, trigger=(1, "epoch")):
        super(HighValueTrigger, self).__init__(
            key, lambda max_value, new_value: new_value > max_value, stock_num, trigger
        )


class LowValueTrigger(BetterValueTrigger):
    def __init__(self, key, stock_num=5, trigger=(1, "epoch")):
        super(LowValueTrigger, self).__init__(
            key, lambda min_value, new_value: new_value < min_value, stock_num, trigger
        )


class PruningExtension(Extension):
    def __init__(
        self,
        trial: optuna.trial.Trial,
        observation_key: str,
        pruner_trigger: Union[
            Tuple[(int, str)], IntervalTrigger, ManualScheduleTrigger
        ],
    ) -> None:
        self._trial = trial
        self._observation_key = observation_key
        self._pruner_trigger = pytorch_trainer.training.get_trigger(pruner_trigger)
        if not isinstance(
            self._pruner_trigger, (IntervalTrigger, ManualScheduleTrigger)
        ):
            pruner_type = type(self._pruner_trigger)
            raise TypeError(
                "Invalid trigger class: " + str(pruner_type) + "\n"
                "Pruner trigger is supposed to be an instance of "
                "IntervalTrigger or ManualScheduleTrigger."
            )

    @staticmethod
    def _get_float_value(observation_value: Union[float, torch.Tensor]) -> float:
        if isinstance(observation_value, torch.Tensor):
            observation_value = observation_value.detach().cpu().numpy()  # type: ignore

        try:
            observation_value = float(observation_value)  # type: ignore
        except TypeError:
            raise TypeError(
                "Type of observation value is not supported by ChainerPruningExtension.\n"
                "{} cannot be cast to float.".format(type(observation_value))
            ) from None

        return observation_value

    def _observation_exists(self, trainer: pytorch_trainer.training.Trainer) -> bool:
        return (
            self._pruner_trigger(trainer)
            and self._observation_key in trainer.observation
        )

    def __call__(self, trainer: pytorch_trainer.training.Trainer) -> None:
        if not self._observation_exists(trainer):
            return

        current_score = self._get_float_value(
            trainer.observation[self._observation_key]
        )
        current_step = getattr(trainer.updater, self._pruner_trigger.unit)
        self._trial.report(current_score, step=current_step)
        if self._trial.should_prune():
            message = "Trial was pruned at {} {}.".format(
                self._pruner_trigger.unit, current_step
            )
            raise optuna.TrialPruned(message)
