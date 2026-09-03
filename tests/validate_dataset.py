import json
from collections import Counter
from pathlib import Path

from config.schema import BenchmarkDataset


DATASET_PATH = Path(__file__).parent.parent / "datasets" / "benchmark_v1.json"


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = BenchmarkDataset(**data)

    print("Dataset schema is valid.")
    print(f"Version: {dataset.version}")
    print(f"Total tests: {len(dataset.tests)}")

    counts = Counter(test.category for test in dataset.tests)

    print("\nCategory counts:")
    for category, count in counts.items():
        print(f"  {category}: {count}")

    assert len(dataset.tests) == 30
    assert counts["factuality"] == 10
    assert counts["relevance"] == 8
    assert counts["format"] == 7
    assert counts["faithfulness"] == 5

    ids = [test.id for test in dataset.tests]

    assert len(ids) == len(set(ids)), "Duplicate test IDs found."

    for test in dataset.tests:
        if test.category == "faithfulness":
            assert test.context is not None, (
                f"{test.id} is a faithfulness test but has no context."
            )

    print("\nAll benchmark validation checks passed.")


if __name__ == "__main__":
    main()