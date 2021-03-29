import argparse
import re
from pathlib import Path
from typing import Optional

import numpy
import yaml
from tqdm import tqdm
from utility.save_arguments import save_arguments
from yukarin_soso.config import Config
from yukarin_soso.dataset import (
    F0ProcessMode,
    FeatureDataset,
    SpeakerFeatureDataset,
    create_dataset,
)
from yukarin_soso.generator import Generator


def _extract_number(f):
    s = re.findall(r"\d+", str(f))
    return int(s[-1]) if s else -1


def _get_predictor_model_path(
    model_dir: Path,
    iteration: int = None,
    prefix: str = "predictor_",
):
    if iteration is None:
        paths = model_dir.glob(prefix + "*.pth")
        model_path = list(sorted(paths, key=_extract_number))[-1]
    else:
        model_path = model_dir / (prefix + "{}.pth".format(iteration))
        assert model_path.exists()
    return model_path


def generate_all(
    model_dir: Path,
    model_iteration: Optional[int],
    model_config: Optional[Path],
    output_dir: Path,
    transpose: bool,
    use_gpu: bool,
):
    if model_config is None:
        model_config = model_dir / "config.yaml"

    output_dir.mkdir(exist_ok=True)
    save_arguments(output_dir / "arguments.yaml", generate_all, locals())

    config = Config.from_dict(yaml.safe_load(model_config.open()))

    model_path = _get_predictor_model_path(
        model_dir=model_dir,
        iteration=model_iteration,
    )
    generator = Generator(
        config=config,
        predictor=model_path,
        use_gpu=use_gpu,
    )

    config.dataset.test_num = 0
    dataset = create_dataset(config.dataset)["train"]

    if isinstance(dataset.dataset, FeatureDataset):
        inputs = dataset.dataset.inputs
        speaker_ids = [None] * len(inputs)
    elif isinstance(dataset.dataset, SpeakerFeatureDataset):
        inputs = dataset.dataset.dataset.inputs
        speaker_ids = dataset.dataset.speaker_ids
    else:
        raise ValueError(dataset)

    for input, speaker_id in tqdm(
        zip(inputs, speaker_ids), total=len(inputs), desc="generate_all"
    ):
        input_data = input.generate()
        data = FeatureDataset.extract_input(
            sampling_length=len(input_data.spec.array),
            f0_data=input_data.f0,
            phoneme_data=input_data.phoneme,
            spec_data=input_data.spec,
            silence_data=input_data.silence,
            phoneme_list_data=input_data.phoneme_list,
            f0_process_mode=F0ProcessMode(config.dataset.f0_process_mode),
            time_mask_max_second=0,
        )

        spec = generator.generate(
            f0=data["f0"][numpy.newaxis],
            phoneme=data["phoneme"][numpy.newaxis],
            speaker_id=(
                numpy.array(speaker_id)[numpy.newaxis]
                if speaker_id is not None
                else None
            ),
        )[0]

        if transpose:
            spec = spec.T

        name = input.f0_path.stem
        numpy.save(output_dir.joinpath(name + ".npy"), spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--model_iteration", type=int)
    parser.add_argument("--model_config", type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--use_gpu", action="store_true")
    generate_all(**vars(parser.parse_args()))
