import json
from pathlib import Path

from config.schema import BenchmarkDataset


DATASETS_DIR = Path(__file__).parent


def load_dataset(version: str) -> BenchmarkDataset:
    dataset_file = DATASETS_DIR / f"{version}.json"

    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {version}"
        )

    with dataset_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return BenchmarkDataset(**data)