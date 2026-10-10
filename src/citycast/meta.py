"""B2: LightGBM P(win) meta-labeler on the gate features (spec §4.3)."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from citycast.decisions import GATE_FEATURES
from citycast.qmodel import PARAMS

META_FEATURES = GATE_FEATURES + ["station", "metric"]


class MetaLabeler:
    def __init__(self, station_categories: tuple[str, ...]):
        self.cats = tuple(station_categories)
        self.model: LGBMClassifier | None = None

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[META_FEATURES].copy()
        X["station"] = pd.Categorical(X["station"], categories=list(self.cats))
        X["metric"] = pd.Categorical(X["metric"], categories=["high", "low"])
        return X

    def fit(self, df: pd.DataFrame) -> "MetaLabeler":
        params = {**PARAMS, "n_estimators": 300}
        self.model = LGBMClassifier(objective="binary", **params).fit(self._X(df), df["won"].to_numpy())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._X(df))[:, 1]


def save(model: MetaLabeler, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(model))
    return path


def load_model(path: Path) -> MetaLabeler:
    return pickle.loads(Path(path).read_bytes())
