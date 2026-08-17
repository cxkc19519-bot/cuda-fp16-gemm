#!/usr/bin/env python3
"""Run a reproducible GEMM benchmark suite and write one consolidated CSV."""

from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def default_binary() -> Path:
    suffix = ".exe" if platform.system() == "Windows" else ""
    candidates = (
        ROOT / "build" / f"gemm_benchmark{suffix}",
        ROOT / "build" / "Release" / f"gemm_benchmark{suffix}",
    )
    return next((path for path in candidates if path.exists()), candidates[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=default_binary())
    parser.add_argument(
        "--suite",
        choices=("default", "llm-decode", "llm-prefill", "llm-all"),
        default="llm-all",
    )
    parser.add_argument("--kernel", default="final-comparison")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "llm_final_comparison.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    binary = args.binary.resolve()
    output = args.output.resolve()
    if not binary.is_file():
        raise SystemExit(f"benchmark executable not found: {binary}")
    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("warmup must be non-negative and iterations must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        "--suite",
        args.suite,
        "--kernel",
        args.kernel,
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--csv",
        str(output),
    ]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        return completed.returncode
    if not output.is_file() or output.stat().st_size == 0:
        raise SystemExit(f"benchmark did not produce CSV: {output}")
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
