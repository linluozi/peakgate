"""Command line: ``python -m peakgate measure|compare module:function``."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any
from typing import Callable

from .core import compare
from .core import measure


def _load(spec: str) -> Callable[[], Any]:
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise SystemExit(f"expected module:function, got {spec!r}")
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    fn = getattr(module, attr, None)
    if not callable(fn):
        raise SystemExit(f"{spec!r} does not resolve to a callable")
    return fn


def _dump(obj: Any, path: str | None) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True)
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"wrote {path}")
    else:
        print(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="peakgate")
    sub = parser.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="measure one callable")
    m.add_argument("target")
    m.add_argument("--reps", type=int, default=31)
    m.add_argument("--warmup", type=int, default=3)
    m.add_argument("--json", default=None)

    c = sub.add_parser("compare", help="compare baseline against candidate")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--reps", type=int, default=31)
    c.add_argument("--warmup", type=int, default=3)
    c.add_argument("--peak-budget", type=float, default=0.01)
    c.add_argument("--time-budget", type=float, default=0.01)
    c.add_argument("--json", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "measure":
        _dump(measure(_load(args.target), reps=args.reps, warmup=args.warmup).as_dict(), args.json)
        return 0
    if args.cmd == "compare":
        result = compare(
            _load(args.baseline),
            _load(args.candidate),
            reps=args.reps,
            warmup=args.warmup,
            peak_budget=args.peak_budget,
            time_budget=args.time_budget,
        )
        _dump(result, args.json)
        return 1 if result["peak_regression"] or result["time_regression"] else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())