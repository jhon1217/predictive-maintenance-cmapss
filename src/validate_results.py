"""Validate published results and fail with a clear message when an invariant changes."""

import json
from pathlib import Path


RESULTS = Path(__file__).resolve().parents[1] / "data" / "results.json"


def main() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    predictions = data["predictions"]
    assert data["source"]["subset"] == "FD001"
    assert metrics["training_rows"] == 20631
    assert metrics["training_engines"] == 100
    assert metrics["test_engines"] == 100
    assert len(predictions) == 100
    assert len({row["engine"] for row in predictions}) == 100
    assert 0 < metrics["official_test_mae"] <= metrics["official_test_rmse"] < 40
    assert 0.5 <= metrics["within_20_cycles"] <= 1
    print("Results validated: 100 test engines and all metric checks passed.")


if __name__ == "__main__":
    main()
