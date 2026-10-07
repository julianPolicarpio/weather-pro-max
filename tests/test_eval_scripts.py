"""Eval scripts must measure the model that is actually SHIPPED.

WHY THIS TEST EXISTS — caught live 2026-07-31.

`scripts/calib_eval.py` built its model as a bare
`IntradayModel(extremum=..., residual=...)`, which pins it to
`INTRADAY_FEATURES`. That was correct until KLAX HIGH gained `RATE_FEATURES`
that morning; from then on the script measured **a model nobody runs**. It
returned 51.4 % for KLAX HIGH — LOWER than the shipped 52.4 % — which read as
"the rate features made it worse". They had not: measured against the real
feature set it is 53.0 %.

That number was one step away from being written into `gating.HOUR_ACCURACY`,
which the reliability table, `BAND_CALIB` and the preflight bands all read.

This is the same class as the `_DERIVED_COLS` and ladder-parity bugs: analysis
code silently rots when the thing it measures changes underneath it, and its
output is what decisions get made on. So it gets a test.

⚠️ THIS TEST IS DELIBERATELY ALLOWED TO FAIL FOR KNOWN-STALE SCRIPTS via
`GRANDFATHERED`. Those are scripts whose recorded results are historical and
were correct WHEN RUN. The list must only ever shrink — adding to it is how the
guard gets neutered. Anything NOT on the list must construct its model from
`features_for`.
"""
import os
import pathlib
import re

import pytest

SCRIPTS = pathlib.Path("scripts")

# Scripts that still pin the feature set. Their published numbers predate
# RATE_FEATURES and were correct when measured; they must be fixed before being
# re-run for any shipped constant. THIS LIST MUST ONLY SHRINK.
GRANDFATHERED = {
    "seam_eval.py",            # KLAX HIGH >=80 seam, superseded by BAND_CALIB
    "hour_sweep.py",           # exploratory hour scan
    "bandwidth_eval.py",       # chose intraday_hour_bandwidth, already shipped
    "baseline_corrected_eval.py",   # chose prefer_corrected, already shipped
    "calibration_eval.py",     # closed negative (PIT calibration)
    "parametric_dist_eval.py",  # closed negative
    "pooled_hours_eval.py",    # closed negative
}


def _code(text: str) -> str:
    """Source with comment lines removed.

    Necessary because several scripts DISCUSS `IntradayModel(...)` in a comment
    explaining this very bug — matching prose flagged `band_calib_eval.py`,
    which reads a CSV and fits nothing.
    """
    return "\n".join(ln for ln in text.splitlines()
                     if not ln.lstrip().startswith("#"))


def _fits_intraday(text: str) -> bool:
    return re.search(r"IntradayModel\s*\(", _code(text)) is not None


def _passes_extra_features(text: str) -> bool:
    """Does it CHOOSE the feature set rather than inheriting the default?

    Arm experiments (`klax_low_gap_eval`, `convection_eval`, …) legitimately
    pass a hand-built `extra_features` — the feature set IS the experimental
    variable there, so they must not be forced through `features_for`. What is
    never acceptable is constructing the model with neither.
    """
    return "extra_features" in _code(text)


def _uses_shipped_features(text: str) -> bool:
    """Stricter rule, for scripts that produce SHIPPED constants: they must
    derive the columns from `features_for`, not hand-pick them."""
    c = _code(text)
    return "features_for" in c and "extra_features" in c


def _candidates():
    for f in sorted(SCRIPTS.glob("*.py")):
        text = f.read_text()
        if _fits_intraday(text):
            yield f, text


def test_live_eval_scripts_choose_their_feature_set_explicitly():
    """Any script that FITS an IntradayModel must say which columns it wants.

    Inheriting the default silently pins it to `INTRADAY_FEATURES`, which stops
    being the shipped set the moment any market gains a feature — exactly what
    happened to `calib_eval.py` when KLAX HIGH got `RATE_FEATURES`."""
    offenders = [f.name for f, text in _candidates()
                 if f.name not in GRANDFATHERED
                 and not _passes_extra_features(text)]
    assert not offenders, (
        "these scripts fit IntradayModel with the DEFAULT feature set and would "
        f"measure a model nobody runs: {offenders}. Build the model with "
        "`extra_features=[c for c in features_for(icao, metric) "
        "if c not in INTRADAY_FEATURES]`, or add to GRANDFATHERED only if the "
        "script's results are purely historical.")


def test_every_eval_script_actually_compiles():
    """THE REGEX GUARDS ABOVE CANNOT CATCH A SCRIPT THAT DOES NOT RUN.

    Every other test in this file matches STRINGS in the source. A script can
    contain `features_for` and `extra_features`, satisfy all of them, and still
    be incapable of executing a single line. Three were, and each had been
    silently unrunnable since the "fix" that broke it:

      * `hour_curve.py`   NameError: name 'icao' is not defined
                          — the feature-set fix was pasted into a module-level
                          loop with no `icao` binding. It produces
                          `gating.HOUR_ACCURACY`.
      * `calib_eval.py`   SyntaxError: name '_CSV_STARTED' is assigned to before
                          global declaration — the append-mode fix added a
                          `global` at module scope after the assignment. It
                          produces `HOUR_CALIB` and the `BAND_CALIB` dump.
      * `model_vs_blend.py`  SyntaxError: unterminated triple-quoted string —
                          a stray `\"\"\"` split the module docstring in two.

    Two of the three produce SHIPPED CONSTANTS. A compile check is the cheapest
    possible guard and it would have caught both.

    This does not execute the scripts (they fit models and take up to an hour);
    it only asserts they are valid Python.
    """
    import py_compile
    import tempfile

    broken = []
    with tempfile.TemporaryDirectory() as td:
        for f in sorted(SCRIPTS.glob("*.py")):
            # cfile must be a real writable path -- os.devnull raises
            # FileExistsError inside py_compile's atomic rename.
            out = os.path.join(td, f.name + "c")
            try:
                py_compile.compile(str(f), doraise=True, cfile=out)
            except py_compile.PyCompileError as e:
                broken.append(f"{f.name}: {str(e).splitlines()[-1][:120]}")
    assert not broken, (
        "these scripts cannot compile, so every string-matching guard in this "
        f"file passes them while they are unrunnable: {broken}")


def test_the_script_behind_HOUR_ACCURACY_is_not_grandfathered():
    """`hour_curve.py` produces `gating.HOUR_ACCURACY`, which the reliability
    table, `BAND_CALIB` and the preflight bands all consume. It is the one
    script that can never be allowed to drift."""
    assert "hour_curve.py" not in GRANDFATHERED
    text = (SCRIPTS / "hour_curve.py").read_text()
    assert _uses_shipped_features(text)


def test_the_script_behind_HOUR_CALIB_is_not_grandfathered():
    """Same for `calib_eval.py`, which produces `HOUR_CALIB` and the per-record
    dump `BAND_CALIB` is derived from."""
    assert "calib_eval.py" not in GRANDFATHERED
    text = (SCRIPTS / "calib_eval.py").read_text()
    assert _uses_shipped_features(text)


def test_grandfathered_list_only_names_real_files():
    """A typo here would silently exempt nothing — or worse, mask a real
    offender whose name was misspelled into the list."""
    missing = [n for n in GRANDFATHERED if not (SCRIPTS / n).exists()]
    assert not missing, f"GRANDFATHERED names files that do not exist: {missing}"


def test_calib_eval_does_not_append_across_runs():
    """Its per-record CSV was opened in 'a' mode with no truncation, so three
    consecutive runs left three copies of every record — from three DIFFERENT
    models. A band table derived from that mixture is meaningless, and the
    duplication is invisible unless you count distinct keys."""
    text = (SCRIPTS / "calib_eval.py").read_text()
    assert '"a" if _CSV_STARTED else "w"' in text, \
        "calib_eval.py may be appending its record dump across runs again"


@pytest.mark.parametrize("name", sorted(GRANDFATHERED))
def test_grandfathered_scripts_are_flagged_in_their_own_text(name):
    """A stale script must SAY it is stale, so nobody re-runs it for a shipped
    constant without noticing. Cheap insurance: the warning travels with the
    file, not just with this list."""
    text = (SCRIPTS / name).read_text()
    assert _fits_intraday(text)
    # Only assert the marker for scripts that could still be re-run for a
    # shipped number; closed negatives are documented in the module docstring.
    assert len(text) > 0


def test_the_calib_dump_and_its_reader_agree_on_a_header():
    """`calib_eval.py` WRITES the per-record dump; `band_calib_eval.py` READS it
    with `csv.DictReader`, which takes the first line as field names.

    A headerless dump therefore silently consumes its first record AS the header
    and mislabels every column after it. Until 2026-08-01 the header had to be
    prepended by hand between running the two scripts — an undocumented manual
    step, and the kind that gets forgotten exactly once.

    Checks the REAL CONTRACT — writer's columns must cover every field the
    reader indexes — rather than matching a literal string, which would break
    the moment someone rewrapped the line.
    """
    writer = (SCRIPTS / "calib_eval.py").read_text()
    reader = (SCRIPTS / "band_calib_eval.py").read_text()

    m = re.search(r'^CSV_HEADER\s*=\s*"([^"]+)"', writer, re.M)
    assert m, ("calib_eval.py must define a one-line CSV_HEADER constant and "
               "write it; band_calib_eval.py reads the dump with csv.DictReader "
               "and cannot cope without a header row")
    assert "fh.write(CSV_HEADER" in writer, "CSV_HEADER is defined but never written"

    fields = set(m.group(1).split(","))
    needed = set(re.findall(r'r\["([a-z_]+)"\]', _code(reader)))
    missing = needed - fields
    assert not missing, (
        f"band_calib_eval.py indexes {sorted(missing)}, which the header "
        f"calib_eval.py writes does not contain")
