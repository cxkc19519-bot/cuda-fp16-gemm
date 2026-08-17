#!/usr/bin/env python3
"""Create dependency-free SVG plots from GEMM benchmark CSV output."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_ORDER = ("reg_vec", "wmma_block", "mma_ptx", "wmma_async", "dispatch", "cublas")
DISPLAY_NAMES = {
    "reg_vec": "CUDA Core RegVec", "wmma_block": "WMMA Block", "mma_ptx": "PTX MMA",
    "wmma_async": "WMMA Async", "dispatch": "Shape-Aware Dispatch", "cublas": "cuBLAS",
}
COLORS = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2", "#111827")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?", default=ROOT / "results" / "llm_final_comparison.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--prefix", default="llm_final")
    parser.add_argument("--title", default="FP16 GEMM: LLM Shape Comparison")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, float | int | str]]:
    required = {"kernel", "M", "N", "K", "latency_us", "tflops", "cublas_ratio"}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise SystemExit(f"CSV is missing required columns: {sorted(required)}")
        rows = [
            {"kernel": row["kernel"], "M": int(row["M"]), "N": int(row["N"]),
             "K": int(row["K"]), "latency_us": float(row["latency_us"]),
             "tflops": float(row["tflops"]), "cublas_ratio": float(row["cublas_ratio"])}
            for row in reader
        ]
    if not rows:
        raise SystemExit(f"CSV contains no data: {path}")
    return rows


def grouped(rows: list[dict[str, float | int | str]]) -> dict[str, list[dict[str, float | int | str]]]:
    result: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        result[str(row["kernel"])].append(row)
    for values in result.values():
        values.sort(key=lambda row: int(row["M"]))
    return result


def svg_plot(data: dict[str, list[dict[str, float | int | str]]], metric: str,
             ylabel: str, title: str) -> str:
    width, height = 1200, 700
    left, right, top, bottom = 95, 250, 70, 80
    plot_width, plot_height = width - left - right, height - top - bottom
    all_rows = [row for values in data.values() for row in values]
    x_values = sorted({int(row["M"]) for row in all_rows})
    multiplier = 100.0 if metric == "cublas_ratio" else 1.0
    y_max = max(float(row[metric]) * multiplier for row in all_rows) * 1.08
    if metric == "cublas_ratio":
        y_max = max(y_max, 108.0)
    x_min_log, x_max_log = math.log2(min(x_values)), math.log2(max(x_values))

    def sx(value: int) -> float:
        return left + (math.log2(value) - x_min_log) / (x_max_log - x_min_log) * plot_width

    def sy(value: float) -> float:
        return top + plot_height - value / y_max * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title)}</title>", '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.tick{font-size:13px}.label{font-size:16px}.title{font-size:23px;font-weight:600}.legend{font-size:14px}.grid{stroke:#d1d5db;stroke-width:1}.axis{stroke:#374151;stroke-width:1.2}</style>',
        f'<text class="title" x="{left}" y="36">{html.escape(title)}</text>',
    ]
    for index in range(7):
        value = y_max * index / 6
        y = sy(value)
        parts.append(f'<line class="grid" x1="{left}" x2="{left + plot_width}" y1="{y:.2f}" y2="{y:.2f}"/>')
        parts.append(f'<text class="tick" text-anchor="end" x="{left - 12}" y="{y + 5:.2f}">{value:.1f}</text>')
    for value in x_values:
        x = sx(value)
        parts.append(f'<line class="grid" x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{top + plot_height}"/>')
        parts.append(f'<text class="tick" text-anchor="middle" x="{x:.2f}" y="{top + plot_height + 28}">{value}</text>')
    parts.extend([
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" class="axis"/>',
        f'<text class="label" text-anchor="middle" x="{left + plot_width / 2}" y="{height - 22}">M (N=K=4096)</text>',
        f'<text class="label" text-anchor="middle" transform="translate(24 {top + plot_height / 2}) rotate(-90)">{html.escape(ylabel)}</text>',
    ])
    if metric == "cublas_ratio":
        reference_y = sy(100.0)
        parts.append(f'<line x1="{left}" x2="{left + plot_width}" y1="{reference_y:.2f}" y2="{reference_y:.2f}" stroke="#6b7280" stroke-dasharray="7 5"/>')
    legend_x, legend_y = left + plot_width + 28, top + 8
    for index, kernel in enumerate(KERNEL_ORDER):
        values = data.get(kernel)
        if not values:
            continue
        color = COLORS[index]
        points = " ".join(f"{sx(int(row['M'])):.2f},{sy(float(row[metric]) * multiplier):.2f}" for row in values)
        line_width = 3.5 if kernel == "dispatch" else 2.0
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{line_width}" stroke-linejoin="round" stroke-linecap="round"/>')
        radius = 4.2 if kernel in ("dispatch", "cublas") else 2.8
        for row in values:
            x, value = sx(int(row["M"])), float(row[metric]) * multiplier
            tooltip = f"{DISPLAY_NAMES.get(kernel, kernel)}: M={row['M']}, {value:.3f}"
            parts.append(f'<circle cx="{x:.2f}" cy="{sy(value):.2f}" r="{radius}" fill="{color}"><title>{html.escape(tooltip)}</title></circle>')
        y_legend = legend_y + index * 34
        parts.append(f'<line x1="{legend_x}" x2="{legend_x + 28}" y1="{y_legend}" y2="{y_legend}" stroke="{color}" stroke-width="{line_width}"/>')
        parts.append(f'<text class="legend" x="{legend_x + 38}" y="{y_legend + 5}">{html.escape(DISPLAY_NAMES.get(kernel, kernel))}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> int:
    args = parse_args()
    data = grouped(load_rows(args.csv.resolve()))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    throughput = output_dir / f"{args.prefix}_tflops.svg"
    ratio = output_dir / f"{args.prefix}_cublas_ratio.svg"
    throughput.write_text(svg_plot(data, "tflops", "Throughput (TFLOPS)", args.title), encoding="utf-8")
    ratio.write_text(svg_plot(data, "cublas_ratio", "Performance vs cuBLAS (%)", args.title), encoding="utf-8")
    print(f"Wrote: {throughput}")
    print(f"Wrote: {ratio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
