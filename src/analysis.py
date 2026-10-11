"""Train and evaluate the FD001 remaining-useful-life model."""

from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "results.json"
NAMES = ["engine", "cycle", "setting_1", "setting_2", "setting_3"] + [
    f"sensor_{index}" for index in range(1, 22)
]


def ensure_data() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    archive_path = RAW / "CMAPSSData.zip"
    if not archive_path.exists():
        urllib.request.urlretrieve(DATA_URL, archive_path)
    required = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
    if not all((RAW / name).exists() for name in required):
        with zipfile.ZipFile(archive_path) as archive:
            for name in required:
                archive.extract(name, RAW)


def read_cycles(filename: str) -> pd.DataFrame:
    return pd.read_csv(RAW / filename, sep=r"\s+", header=None, names=NAMES)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    sensors = [name for name in NAMES if name.startswith("sensor_")]
    grouped = result.groupby("engine", sort=False)
    for sensor in sensors:
        result[f"{sensor}_delta"] = grouped[sensor].diff().fillna(0)
        result[f"{sensor}_mean5"] = grouped[sensor].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    return result


def main() -> None:
    ensure_data()
    train = read_cycles("train_FD001.txt")
    test = read_cycles("test_FD001.txt")
    truth = pd.read_csv(RAW / "RUL_FD001.txt", sep=r"\s+", header=None, names=["rul"])
    truth["engine"] = np.arange(1, len(truth) + 1)

    train["rul"] = (train.groupby("engine")["cycle"].transform("max") - train["cycle"]).clip(upper=125)
    raw_sensors = [name for name in NAMES if name.startswith("sensor_")]
    useful = [name for name in raw_sensors if train[name].std() > 1e-6]
    selected = train[useful].corrwith(train["rul"], method="spearman").abs().nlargest(12).index.tolist()

    train_features = add_features(train)
    test_features = add_features(test)
    features = ["cycle", "setting_1", "setting_2", "setting_3"]
    for sensor in selected:
        features.extend([sensor, f"{sensor}_delta", f"{sensor}_mean5"])

    development = train_features[train_features["engine"] <= 80]
    validation = train_features[train_features["engine"] > 80]
    settings = dict(max_depth=16, min_samples_leaf=3, max_features=0.75, n_jobs=-1, random_state=2026)
    validation_model = RandomForestRegressor(n_estimators=220, **settings).fit(development[features], development["rul"])
    validation_prediction = validation_model.predict(validation[features])

    model = RandomForestRegressor(n_estimators=300, **settings).fit(train_features[features], train_features["rul"])
    last_rows = test_features.loc[test_features.groupby("engine")["cycle"].idxmax()].copy()
    last_rows["actual_rul"] = truth.set_index("engine").loc[last_rows["engine"], "rul"].to_numpy()
    last_rows["predicted_rul"] = np.clip(model.predict(last_rows[features]), 0, 125)
    last_rows["absolute_error"] = (last_rows["predicted_rul"] - last_rows["actual_rul"]).abs()

    feature_importance = pd.Series(model.feature_importances_, index=features)
    grouped_importance: dict[str, float] = {}
    for feature, value in feature_importance.items():
        root = feature.split("_delta")[0].split("_mean5")[0]
        grouped_importance[root] = grouped_importance.get(root, 0.0) + float(value)
    importance = sorted(grouped_importance.items(), key=lambda item: item[1], reverse=True)[:8]

    output = {
        "source": {"name": "NASA C-MAPSS Jet Engine Simulated Data", "url": "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data", "subset": "FD001"},
        "metrics": {
            "training_rows": int(len(train)), "training_engines": int(train.engine.nunique()), "test_engines": int(test.engine.nunique()),
            "validation_rmse": round(float(mean_squared_error(validation.rul, validation_prediction) ** 0.5), 2),
            "validation_mae": round(float(mean_absolute_error(validation.rul, validation_prediction)), 2),
            "validation_r2": round(float(r2_score(validation.rul, validation_prediction)), 3),
            "official_test_rmse": round(float(mean_squared_error(last_rows.actual_rul, last_rows.predicted_rul) ** 0.5), 2),
            "official_test_mae": round(float(mean_absolute_error(last_rows.actual_rul, last_rows.predicted_rul)), 2),
            "within_20_cycles": round(float((last_rows.absolute_error <= 20).mean()), 3),
        },
        "predictions": [{"engine": int(row.engine), "actual": round(float(row.actual_rul), 1), "predicted": round(float(row.predicted_rul), 1), "error": round(float(row.absolute_error), 1)} for row in last_rows.itertuples()],
        "importance": [{"feature": name.replace("_", " ").title(), "value": round(value, 4)} for name, value in importance],
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["metrics"], indent=2))


if __name__ == "__main__":
    main()
