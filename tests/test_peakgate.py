import json
import os
import subprocess
import sys

import pytest

from peakgate.core import _stats
from peakgate.core import compare
from peakgate.core import measure
from peakgate.core import required_reps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _alloc(n):
    return [0] * n


def test_stats_quartiles():
    s = _stats([1, 2, 3, 4, 5])
    assert s.n == 5
    assert s.median == 3.0
    assert s.p25 == 2.0
    assert s.p75 == 4.0
    assert s.min == 1.0 and s.max == 5.0
    assert s.spread == 4.0
    assert s.distinct == 5


def test_single_sample_stats():
    s = _stats([7])
    assert s.median == 7.0 and s.p25 == 7.0 and s.p75 == 7.0
    assert s.rel_sd == 0.0


def test_measure_returns_reps_samples():
    m = measure(lambda: [0] * 1000, reps=5, warmup=1)
    assert m.peak.n == 5 and m.time_ms.n == 5
    assert len(m.peaks) == 5 and len(m.times_ms) == 5


def test_measure_detects_allocation_difference():
    small = measure(lambda: [0] * 100, reps=5, warmup=1)
    big = measure(lambda: [0] * 100_000, reps=5, warmup=1)
    assert big.peak.median > small.peak.median * 100


def test_peak_is_nearly_deterministic():
    m = measure(lambda: [0] * 100_000, reps=10, warmup=1)
    # only the measurement call itself (a tuple) jitters, by a few bytes
    assert m.peak.spread < 256, m.peak.spread
    assert m.peak.distinct <= 3, m.peak.distinct


def test_compare_flags_peak_regression():
    r = compare(lambda: [0] * 1000, lambda: [0] * 2000, reps=5, warmup=1,
                peak_budget=0.01, time_budget=10.0)
    assert r["peak_ratio"] > 1.05
    assert r["peak_regression"] is True
    assert r["time_regression"] is False


def test_compare_identical_has_no_peak_regression():
    r = compare(lambda: [0] * 1000, lambda: [0] * 1000, reps=5, warmup=1,
                peak_budget=0.01, time_budget=10.0)
    assert abs(r["peak_ratio"] - 1.0) < 0.01
    assert r["peak_regression"] is False


def test_exception_propagates():
    def boom():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        measure(boom, reps=1, warmup=0)


def test_validation():
    with pytest.raises(ValueError):
        measure(lambda: None, reps=0)
    with pytest.raises(ValueError):
        measure(lambda: None, warmup=-1)
    with pytest.raises(ValueError):
        required_reps(0.02, 0.0)


def test_required_reps_formula():
    assert required_reps(0.0, 0.01) == 1
    # 2% noise, 1% effect, 95% two-sided: 2 * (1.96*2)**2 ~= 31
    assert required_reps(0.02, 0.01) == 31
    assert required_reps(0.02, 0.02) == 8


def test_cli_measure_and_compare(tmp_path):
    (tmp_path / "w.py").write_text(
        "def a():\n    return [0] * 100\n\ndef b():\n    return [0] * 200\n",
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=ROOT)
    out = tmp_path / "m.json"
    r = subprocess.run(
        [sys.executable, "-m", "peakgate", "measure", "w:a", "--reps", "3",
         "--warmup", "0", "--json", str(out)],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text())
    assert data["peak"]["n"] == 3
    r = subprocess.run(
        [sys.executable, "-m", "peakgate", "compare", "w:a", "w:a", "--reps", "3",
         "--warmup", "0", "--peak-budget", "0.5"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    r = subprocess.run(
        [sys.executable, "-m", "peakgate", "compare", "w:a", "w:b", "--reps", "3",
         "--warmup", "0", "--peak-budget", "0.01", "--time-budget", "100"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert r.returncode == 1, r.stdout