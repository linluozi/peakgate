"""Produce evidence/summary.json and evidence/raw/*.json.

Run from the repository root: ``python3 evidence/run_evidence.py``.
Corpus: 1,000,000 characters of real CPython standard-library source (sorted
paths, first N files) -- real text, not synthetic filler.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import platform
import re
import statistics
import sys
import sysconfig
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from peakgate.core import compare  # noqa: E402
from peakgate.core import measure  # noqa: E402
from peakgate.core import required_reps  # noqa: E402

RAW = os.path.join(ROOT, "evidence", "raw")
os.makedirs(RAW, exist_ok=True)


def build_corpus(limit: int = 1_000_000) -> tuple[str, dict]:
    stdlib = sysconfig.get_paths()["stdlib"]
    paths = sorted(glob.glob(os.path.join(stdlib, "*.py")))
    chunks, files = [], []
    total = 0
    for p in paths:
        try:
            text = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        files.append(os.path.basename(p))
        chunks.append(text)
        total += len(text)
        if total >= limit:
            break
    corpus = "".join(chunks)[:limit]
    meta = {
        "stdlib": stdlib,
        "files": len(files),
        "first_files": files[:5],
        "chars": len(corpus),
        "sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
    }
    return corpus, meta


def dump(name: str, obj) -> None:
    with open(os.path.join(RAW, name), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> int:
    corpus, meta = build_corpus()
    word = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
    payload = [{"i": i, "w": w} for i, w in enumerate(corpus.split()[:5000])]

    def findall():
        return word.findall(corpus)

    def finditer():
        return sum(1 for _ in word.finditer(corpus))

    def dumps_payload():
        return json.dumps(payload)

    workloads = {"findall": findall, "finditer": finditer, "json_dumps": dumps_payload}

    def cpu_model() -> str:
        try:
            for line in open("/proc/cpuinfo", encoding="utf-8"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return platform.processor()

    def mem_total_kb() -> int:
        try:
            for line in open("/proc/meminfo", encoding="utf-8"):
                if line.startswith("MemTotal:"):
                    return int(line.split()[1])
        except OSError:
            pass
        return 0

    summary: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": cpu_model(),
        "mem_total_kb": mem_total_kb(),
        "corpus": meta,
        "determinism": {},
        "aa": {},
        "real_case": {},
        "sensitivity": {},
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    for name, fn in workloads.items():
        m = measure(fn, reps=40, warmup=3)
        dump(f"determinism-{name}.json", m.as_dict())
        summary["determinism"][name] = {
            "peak_median": m.peak.median,
            "peak_spread_bytes": m.peak.spread,
            "peak_distinct_in_40": m.peak.distinct,
            "peak_rel_sd": m.peak.rel_sd,
            "time_ms_median": m.time_ms.median,
            "time_rel_sd": m.time_ms.rel_sd,
            "required_reps_1pct": required_reps(m.time_ms.rel_sd, 0.01),
        }

    # A/A: the same callable in both arms, 30 independent 3-rep trials.
    # Any "regression" at a 1% budget is a false positive.
    def cheap():
        return [0] * 1000

    peak_fp = time_fp = trials = 0
    ratios_peak, ratios_time = [], []
    for trial in range(30):
        r = compare(cheap, cheap, reps=3, warmup=1, peak_budget=0.01, time_budget=0.01)
        trials += 1
        peak_fp += bool(r["peak_regression"])
        time_fp += bool(r["time_regression"])
        ratios_peak.append(r["peak_ratio"])
        ratios_time.append(r["time_ratio"])
        dump(f"aa-{trial:02d}.json", r)
    summary["aa"] = {
        "trials": trials,
        "reps_per_arm": 3,
        "peak_budget": 0.01,
        "time_budget": 0.01,
        "peak_false_positives": peak_fp,
        "time_false_positives": time_fp,
        "peak_ratio_min": min(ratios_peak),
        "peak_ratio_max": max(ratios_peak),
        "time_ratio_min": min(ratios_time),
        "time_ratio_max": max(ratios_time),
    }

    # Real corpus case: materialise all matches vs stream them.
    r = compare(findall, finditer, reps=15, warmup=3, peak_budget=0.01, time_budget=0.01)
    dump("real-findall-vs-finditer.json", r)
    summary["real_case"] = {
        "case": "re.findall vs re.finditer over 1 MB of real stdlib source",
        "matches": len(findall()),
        "peak_baseline_bytes": r["baseline"]["peak"]["median"],
        "peak_candidate_bytes": r["candidate"]["peak"]["median"],
        "peak_ratio": r["peak_ratio"],
        "time_ratio": r["time_ratio"],
    }

    # Sensitivity: a 2% allocation difference between two otherwise equal
    # callables.  How many samples does each axis need?
    def base_alloc():
        return [0] * 1_000_000

    def plus2pct():
        return [0] * 1_020_000

    r2 = compare(base_alloc, plus2pct, reps=5, warmup=1, peak_budget=0.005, time_budget=0.005)
    dump("sensitivity-2pct.json", r2)
    summary["sensitivity"] = {
        "case": "identical loops allocating 1,000,000 vs 1,020,000 ints (2%)",
        "measured_peak_ratio": r2["peak_ratio"],
        "peak_regression_detected_at_5_reps": r2["peak_regression"],
        "time_ratio": r2["time_ratio"],
        "time_regression_at_5_reps": r2["time_regression"],
        "time_rel_sd_for_2pct": r2["candidate"]["time_ms"]["rel_sd"],
        "reps_time_needs_for_2pct": required_reps(r2["candidate"]["time_ms"]["rel_sd"], 0.02),
    }

    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(ROOT, "evidence", "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())