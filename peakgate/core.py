"""Peak-allocation measurement and an A/B gate built on it.

The unit of measurement is *peak traced bytes allocated above the starting
high-water mark* for one call.  ``tracemalloc.reset_peak()`` makes each call
start from its own baseline, so the number is a per-call allocation peak rather
than a process high-water mark.  The value is nearly deterministic (the only
jitter is the few bytes allocated by ``get_traced_memory()`` itself), which is
what makes a one-sample allocation gate possible.
"""

from __future__ import annotations

import gc
import statistics
import time
import tracemalloc
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Callable


@dataclass(frozen=True)
class Stats:
    """Summary of one sample vector.  All percentiles use linear interpolation."""

    n: int
    median: float
    p25: float
    p75: float
    min: float
    max: float
    spread: float
    rel_sd: float
    distinct: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Measurement:
    peak: Stats
    time_ms: Stats
    peaks: tuple[int, ...]
    times_ms: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "peak": self.peak.as_dict(),
            "time_ms": self.time_ms.as_dict(),
            "peaks": list(self.peaks),
            "times_ms": list(self.times_ms),
        }


def _pct(sorted_xs: list[float], q: float) -> float:
    if len(sorted_xs) == 1:
        return float(sorted_xs[0])
    pos = (len(sorted_xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = pos - lo
    return float(sorted_xs[lo] * (1.0 - frac) + sorted_xs[hi] * frac)


def _stats(xs: list[float] | list[int]) -> Stats:
    s = sorted(float(x) for x in xs)
    med = _pct(s, 0.5)
    mean = statistics.fmean(s)
    sd = statistics.stdev(s) if len(s) > 1 else 0.0
    return Stats(
        n=len(s),
        median=med,
        p25=_pct(s, 0.25),
        p75=_pct(s, 0.75),
        min=s[0],
        max=s[-1],
        spread=s[-1] - s[0],
        rel_sd=(sd / mean) if mean else 0.0,
        distinct=len(set(xs)),
    )


def _one(fn: Callable[[], Any], collect: bool) -> tuple[int, float]:
    if collect:
        gc.collect()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    peak = tracemalloc.get_traced_memory()[1]
    del out
    return max(0, peak - base), (t1 - t0) * 1000.0


def measure(
    fn: Callable[[], Any],
    *,
    reps: int = 31,
    warmup: int = 3,
    collect: bool = True,
) -> Measurement:
    """Run ``fn`` ``reps`` times, returning per-call peak bytes and wall time.

    ``warmup`` calls are executed first and discarded (imports, caches and code
    specialisation are not part of the measured steady state).  ``collect``
    runs ``gc.collect()`` before every sample so a previous call's garbage is
    not attributed to the next call.
    """
    if reps < 1:
        raise ValueError("reps must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    owned = not tracemalloc.is_tracing()
    if owned:
        tracemalloc.start()
    try:
        for _ in range(warmup):
            fn()
        peaks: list[int] = []
        times: list[float] = []
        for _ in range(reps):
            p, t = _one(fn, collect)
            peaks.append(p)
            times.append(t)
    finally:
        if owned:
            tracemalloc.stop()

    return Measurement(
        peak=_stats(peaks),
        time_ms=_stats(times),
        peaks=tuple(peaks),
        times_ms=tuple(times),
    )


def compare(
    baseline: Callable[[], Any],
    candidate: Callable[[], Any],
    *,
    reps: int = 31,
    warmup: int = 3,
    peak_budget: float = 0.01,
    time_budget: float = 0.01,
    collect: bool = True,
) -> dict[str, Any]:
    """Interleaved A/B comparison of peak bytes and wall time.

    Arms alternate order each repetition so slow machine drift does not
    systematically favour one side.  ``*_regression`` answers "did the
    candidate exceed the baseline by more than the budget?"; ratios below 1.0
    mean the candidate used less / was faster.
    """
    if reps < 1:
        raise ValueError("reps must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    owned = not tracemalloc.is_tracing()
    if owned:
        tracemalloc.start()
    bp: list[int] = []
    bt: list[float] = []
    cp: list[int] = []
    ct: list[float] = []
    try:
        for _ in range(warmup):
            baseline()
            candidate()
        for i in range(reps):
            order = (
                ((baseline, bp, bt), (candidate, cp, ct))
                if i % 2 == 0
                else ((candidate, cp, ct), (baseline, bp, bt))
            )
            for fn, ps, ts in order:
                p, t = _one(fn, collect)
                ps.append(p)
                ts.append(t)
    finally:
        if owned:
            tracemalloc.stop()

    b_peak = _stats(bp)
    c_peak = _stats(cp)
    b_time = _stats(bt)
    c_time = _stats(ct)
    peak_ratio = _ratio(c_peak.median, b_peak.median)
    time_ratio = _ratio(c_time.median, b_time.median)

    return {
        "reps": reps,
        "peak_budget": peak_budget,
        "time_budget": time_budget,
        "baseline": {"peak": b_peak.as_dict(), "time_ms": b_time.as_dict()},
        "candidate": {"peak": c_peak.as_dict(), "time_ms": c_time.as_dict()},
        "peak_ratio": peak_ratio,
        "time_ratio": time_ratio,
        "peak_regression": peak_ratio > 1.0 + peak_budget,
        "time_regression": time_ratio > 1.0 + time_budget,
        "peak_samples": {"baseline": bp, "candidate": cp},
        "time_samples": {"baseline": bt, "candidate": ct},
    }


def _ratio(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return 1.0 if candidate == 0 else float("inf")
    return candidate / baseline


def required_reps(rel_sd: float, effect: float, z: float = 1.96) -> int:
    """Reps needed for a two-sided mean test to resolve ``effect`` (relative).

    Uses the normal approximation ``n = 2 * (z * rel_sd / effect) ** 2`` and
    rounds up.  This is the number that makes the time axis comparable to the
    allocation axis, which needs one sample for the same effect.
    """
    if effect <= 0:
        raise ValueError("effect must be > 0")
    if rel_sd <= 0:
        return 1
    return max(1, int(round(2.0 * (z * rel_sd / effect) ** 2 + 0.5)))