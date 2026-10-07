import pytest
from sandbox_helpers import make_market

from sandbox.pricing import devig, mid, multiclass_brier


def test_mid_uses_whatever_side_exists():
    assert mid(make_market("A", bid=0.20, ask=0.24)) == pytest.approx(0.22)
    assert mid(make_market("A", bid=0.0, ask=0.30)) == pytest.approx(0.30)
    assert mid(make_market("A", bid=0.40, ask=1.0)) == pytest.approx(0.40)
    assert mid(make_market("A", bid=0.0, ask=1.0)) is None


def test_devig_normalises_the_ladder():
    ms = [make_market("A", bid=0.20, ask=0.24), make_market("B", bid=0.50, ask=0.56),
          make_market("C", bid=0.0, ask=0.30)]
    p = devig(ms)
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["A"] == pytest.approx(0.22 / 1.05)
    assert p["B"] == pytest.approx(0.53 / 1.05)


def test_multiclass_brier():
    assert multiclass_brier({"A": 0.2, "B": 0.5, "C": 0.3}, "B") == pytest.approx(0.38)
    assert multiclass_brier({"A": 1.0}, "B") == pytest.approx(2.0)  # winner missing counts as 0
