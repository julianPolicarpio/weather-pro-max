import numpy as np
import pandas as pd

from citycast.decisions import GATE_FEATURES
from citycast.meta import MetaLabeler, load_model, save


def _frame(seed: int, n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in GATE_FEATURES})
    df["p_model"] = rng.uniform(0, 1, n)
    df["won"] = (rng.uniform(0, 1, n) < df["p_model"]).astype(int)
    df["station"], df["metric"] = rng.choice(["KNYC", "KLAX"], n), "high"
    return df


def test_learns_signal_and_round_trips(tmp_path):
    train, held_out = _frame(0), _frame(1)
    m = MetaLabeler(("KLAX", "KNYC")).fit(train)
    p = m.predict(held_out)
    assert p.shape == (len(held_out),) and 0 <= p.min() and p.max() <= 1
    assert np.corrcoef(p, held_out["p_model"])[0, 1] > 0.8
    save(m, tmp_path / "b2.pkl")
    assert np.allclose(load_model(tmp_path / "b2.pkl").predict(held_out), p)
