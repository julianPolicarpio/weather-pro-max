from contextlib import closing
from datetime import date

from citycast import dataset, db
from citycast.features import FEATURE_COLS, FeatureContext
from test_citycast_features import NYC, _seed


def test_build_one_day_has_grid_rows_and_truth(tmp_path, monkeypatch):
    with closing(db.connect(tmp_path / "h.db")) as c:
        _seed(c)
        with c:
            db.upsert(c, "cli_daily", [{"station": "KNYC", "date": "2024-01-05", "high_f": 47,
                                        "low_f": 30, "final": 1, "issued_utc": None, "raw": "{}",
                                        "fetched_at": "x"}])
        ctx = FeatureContext(c, {"KNYC": NYC}, tmp_path / "h.db")
        df = dataset.build(ctx, "high", date(2024, 1, 5), date(2024, 1, 5), progress=lambda m: None)
        # 14 grid points; NBS (13Z Jan 4, 3 h lag) is usable from 16Z and MEX (12Z, 5 h lag) from 17Z,
        # so 06:00 and 09:00 EST on D-1 (11Z, 14Z) have no anchor
        assert len(df) == 12 and set(FEATURE_COLS) <= set(df.columns)
        assert (df["y"] == 47.0).all() and df["asof_utc"].is_monotonic_increasing
        monkeypatch.setattr(dataset, "DATASET_DIR", tmp_path / "ds")
        dataset.save(df, "high", "fp-1")
        assert dataset.load("high").equals(df)


def test_save_stamps_the_given_fingerprint_and_load_keeps_it(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setattr(dataset, "DATASET_DIR", tmp_path / "ds")
    dataset.save(pd.DataFrame({"station": ["KNYC"], "y": [47.0]}), "high", "fp-from-the-caller")
    assert dataset.load("high").attrs["fingerprint"] == "fp-from-the-caller"
    assert not hasattr(dataset, "code_fingerprint")                  # save never hashes the code itself


def test_dataset_command_stamps_the_hash_taken_at_command_start(tmp_path, monkeypatch, capsys):
    import json

    import pandas as pd

    from citycast import cli, fingerprint
    calls = []

    def hashes():                                                    # the code "changes" after the first hash
        calls.append(1)
        return f"hash-{len(calls)}"
    monkeypatch.setattr(fingerprint, "code_fingerprint", hashes)
    monkeypatch.setattr(cli, "code_fingerprint", hashes)
    monkeypatch.setattr(dataset, "DATASET_DIR", tmp_path / "ds")
    monkeypatch.setattr(cli, "FeatureContext", lambda *a, **k: object())

    def build(ctx, metric, since, until, icaos=None, progress=print):
        assert len(calls) == 1                                       # hashed before the (long) build started
        return pd.DataFrame({"station": ["KNYC"], "y": [47.0]})
    monkeypatch.setattr(cli.dataset, "build", build)
    assert cli.main(["--db", str(tmp_path / "h.db"), "dataset", "--metric", "high"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["rows"] == 1
    assert dataset.load("high").attrs["fingerprint"] == "hash-1"
    assert len(calls) == 1
