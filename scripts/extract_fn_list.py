import argparse
from glob import glob
from pathlib import Path

import numpy
import yaml
from yukarin_soso.config import Config


def extract_fn_list(
    config_yaml_path: Path,
    output_train_path: Path,
    output_test_path: Path,
):
    with config_yaml_path.open() as f:
        config = Config.from_dict(yaml.safe_load(f)).dataset

    f0_paths = {Path(p).stem: Path(p) for p in glob(config.f0_glob)}
    fn_list = sorted(f0_paths.keys())
    assert len(fn_list) > 0

    numpy.random.RandomState(config.seed).shuffle(fn_list)

    test_num = config.test_num
    trains = fn_list[test_num:]
    tests = fn_list[:test_num]

    output_train_path.write_text("\n".join(sorted(trains)))
    output_test_path.write_text("\n".join(sorted(tests)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml_path", type=Path)
    parser.add_argument(
        "--output_train_path", type=Path, default=Path("/tmp/train_fn_list.txt")
    )
    parser.add_argument(
        "--output_test_path", type=Path, default=Path("/tmp/test_fn_list.txt")
    )
    extract_fn_list(**vars(parser.parse_args()))
