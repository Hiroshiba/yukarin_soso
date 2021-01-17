import argparse
import re
from pathlib import Path
from typing import Optional

import numpy
import yaml
from more_itertools import chunked
from pytorch_trainer.dataset.convert import concat_examples
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm
from utility.save_arguments import save_arguments
from yukarin_soso.config import Config
from yukarin_soso.dataset import (
    FeatureDataset,
    SpeakerFeatureDataset,
    TensorWrapperDataset,
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


def generate(
    model_dir: Path,
    model_iteration: Optional[int],
    model_config: Optional[Path],
    time_second: float,
    num_test: int,
    output_dir: Path,
    use_gpu: bool,
):
    if model_config is None:
        model_config = model_dir / "config.yaml"

    output_dir.mkdir(exist_ok=True)
    save_arguments(output_dir / "arguments.yaml", generate, locals())

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

    sampling_rate = 24000 / 512
    config.dataset.sampling_length = int(sampling_rate * time_second)

    batch_size = config.train.batch_size

    dataset = create_dataset(config.dataset)["test"]
    if isinstance(dataset, ConcatDataset):
        dataset = dataset.datasets[0]

    if isinstance(dataset.dataset, FeatureDataset):
        f0_paths = [inp.f0_path for inp in dataset.dataset.inputs[:num_test]]
    elif isinstance(dataset.dataset, SpeakerFeatureDataset):
        f0_paths = [inp.f0_path for inp in dataset.dataset.dataset.inputs[:num_test]]
    else:
        raise ValueError(dataset)

    for data, f0_path in zip(
        chunked(tqdm(dataset, desc="generate"), batch_size),
        chunked(f0_paths, batch_size),
    ):
        data = concat_examples(data)
        specs = generator.generate(
            f0=data["f0"],
            phoneme=data["phoneme"],
            speaker_id=data["speaker_id"] if "speaker_id" in data else None,
        )

        for spec, p in zip(specs, f0_path):
            numpy.save(output_dir.joinpath(p.stem + ".npy"), spec)

    # validation
    # for input, speaker_id in zip(
    #     chunked(tqdm(inputs, desc="generate"), batch_size),
    #     chunked(speaker_ids, batch_size),
    # ):
    #     # padding
    #     f0_datas = [SamplingData.load(inp.f0_path) for inp in input]
    #     f0 = numpy.stack(
    #         [d.array[: int(time_length * f0_datas[0].rate)] for d in f0_datas]
    #     )

    #     phoneme_datas = [SamplingData.load(inp.phoneme_path) for inp in input]
    #     phoneme = numpy.stack(
    #         [d.array[: int(time_length * phoneme_datas[0].rate)] for d in phoneme_datas]
    #     )

    #     # generate
    #     specs = generator.generate(
    #         f0=f0,
    #         phoneme=phoneme,
    #         speaker_id=speaker_id if speaker_id[0] is not None else None,
    #     )

    #     # save
    #     for spec, inp in zip(specs, input):
    #         numpy.save(output_dir.joinpath(inp.f0_path.stem + ".npy"), spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=Path)
    parser.add_argument("--model_iteration", type=int)
    parser.add_argument("--model_config", type=Path)
    parser.add_argument("--time_second", type=float, default=3)
    parser.add_argument("--num_test", type=int, default=10)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--use_gpu", action="store_true")
    generate(**vars(parser.parse_args()))
