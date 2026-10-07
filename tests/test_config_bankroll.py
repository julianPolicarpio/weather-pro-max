import shutil

import pytest

from highcast.config import MAX_BANKROLL, load_config, set_bankroll


def _copy_cfg(tmp_path):
    dst = tmp_path / "klax.yaml"
    shutil.copy("config/klax.yaml", dst)
    return dst


def test_set_bankroll_updates_value(tmp_path):
    p = _copy_cfg(tmp_path)
    out = set_bankroll(p, 250)
    assert out == 250.0
    assert load_config(p).staking.bankroll == 250.0


def test_set_bankroll_preserves_comment_and_other_keys(tmp_path):
    p = _copy_cfg(tmp_path)
    before = p.read_text().splitlines()
    set_bankroll(p, 250)
    after = p.read_text().splitlines()
    diff = [(b, a) for b, a in zip(before, after) if b != a]
    assert len(diff) == 1, "exactly one line should change"
    b, a = diff[0]
    assert "bankroll" in b and "bankroll" in a
    # the trailing comment text (everything after the first '#') survives verbatim
    assert "#" in a and a.split("#", 1)[1] == b.split("#", 1)[1]
    cfg = load_config(p)
    assert cfg.staking.kelly_fraction == 0.25
    assert cfg.staking.currency == "USD"


def test_set_bankroll_zero_is_valid(tmp_path):
    p = _copy_cfg(tmp_path)
    set_bankroll(p, 100)
    assert set_bankroll(p, 0) == 0.0
    assert load_config(p).staking.bankroll == 0.0


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), MAX_BANKROLL * 2])
def test_set_bankroll_rejects_bad_values(tmp_path, bad):
    p = _copy_cfg(tmp_path)
    set_bankroll(p, 0)  # normalize: don't depend on the live config's bankroll value
    with pytest.raises(ValueError):
        set_bankroll(p, bad)
    assert load_config(p).staking.bankroll == 0.0  # file untouched on rejection


def test_set_bankroll_missing_line_raises(tmp_path):
    p = tmp_path / "nobank.yaml"
    p.write_text("staking:\n  kelly_fraction: 0.25\n")
    with pytest.raises(ValueError):
        set_bankroll(p, 100)
