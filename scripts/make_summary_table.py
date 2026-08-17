#!/usr/bin/env python3
"""Generate an exact SVG benchmark summary table from per-shape CSV files."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


KERNELS = ("reg_vec", "wmma_block", "wmma_async", "mma_ptx", "cublas")
HEADERS = ("Shape", "RegVec", "WMMA Block", "WMMA Async", "MMA PTX", "cuBLAS", "Best/cuBLAS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--title", default="RTX 4090 (sm_89) FP16 GEMM Performance")
    return parser.parse_args()


def read_shape(path: Path) -> dict[str, str | float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_kernel = {row["kernel"]: row for row in rows}
    missing = set(KERNELS) - set(by_kernel)
    if missing:
        raise SystemExit(f"{path} is missing kernels: {sorted(missing)}")
    first = rows[0]
    values = {kernel: float(by_kernel[kernel]["tflops"]) for kernel in KERNELS}
    best = max(values[kernel] for kernel in KERNELS if kernel != "cublas")
    values.update(
        {
            "shape": f"{first['M']}x{first['N']}x{first['K']}",
            "best_ratio": best / values["cublas"] * 100.0,
        }
    )
    return values


def read_summary(path: Path) -> list[dict[str, str | float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"shape", *KERNELS, "best_cublas_ratio"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"invalid summary CSV: {path}")
    return [
        {
            "shape": row["shape"],
            **{kernel: float(row[kernel]) for kernel in KERNELS},
            "best_ratio": float(row["best_cublas_ratio"]),
        }
        for row in rows
    ]


def write_summary_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("shape", *KERNELS, "best_cublas_ratio"))
        for row in rows:
            writer.writerow(
                (row["shape"], *(f"{float(row[kernel]):.3f}" for kernel in KERNELS),
                 f"{float(row['best_ratio']):.3f}")
            )


def make_svg(title: str, rows: list[dict[str, str | float]]) -> str:
    widths = (180, 110, 145, 150, 125, 110, 145)
    left, top, title_height, row_height = 20, 18, 52, 52
    table_width = sum(widths)
    width = table_width + left * 2
    height = top + title_height + row_height * (len(rows) + 1) + 20
    table_top = top + title_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title)}</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.title{font-size:24px;font-weight:700}.header{font-size:16px;font-weight:700}.cell{font-size:15px}.grid{stroke:#cbd5e1;stroke-width:1}</style>',
        f'<text class="title" x="{left}" y="{top + 28}">{html.escape(title)}</text>',
    ]
    for row_index in range(len(rows) + 1):
        y = table_top + row_index * row_height
        fill = "#ffffff" if row_index == 0 or row_index % 2 == 0 else "#f3f6f9"
        parts.append(f'<rect x="{left}" y="{y}" width="{table_width}" height="{row_height}" fill="{fill}"/>')
    x = left
    for column, column_width in enumerate(widths):
        parts.append(f'<rect class="grid" x="{x}" y="{table_top}" width="{column_width}" height="{row_height * (len(rows) + 1)}" fill="none"/>')
        parts.append(f'<text class="header" text-anchor="middle" x="{x + column_width / 2}" y="{table_top + 32}">{html.escape(HEADERS[column])}</text>')
        x += column_width
    for row_index, row in enumerate(rows, start=1):
        values = (
            str(row["shape"]), f"{float(row['reg_vec']):.3f}", f"{float(row['wmma_block']):.3f}",
            f"{float(row['wmma_async']):.3f}", f"{float(row['mma_ptx']):.3f}",
            f"{float(row['cublas']):.3f}", f"{float(row['best_ratio']):.1f}%",
        )
        x = left
        y = table_top + row_index * row_height
        for value, column_width in zip(values, widths):
            parts.append(f'<line class="grid" x1="{x}" x2="{x + column_width}" y1="{y}" y2="{y}"/>')
            parts.append(f'<text class="cell" text-anchor="middle" x="{x + column_width / 2}" y="{y + 32}">{html.escape(value)}</text>')
            x += column_width
    bottom = table_top + row_height * (len(rows) + 1)
    parts.append(f'<line class="grid" x1="{left}" x2="{left + table_width}" y1="{bottom}" y2="{bottom}"/>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> int:
    args = parse_args()
    if len(args.inputs) == 1:
        with args.inputs[0].open(newline="", encoding="utf-8") as stream:
            fields = csv.DictReader(stream).fieldnames or []
        rows = read_summary(args.inputs[0].resolve()) if "shape" in fields else [read_shape(args.inputs[0].resolve())]
    else:
        rows = [read_shape(path.resolve()) for path in args.inputs]
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.svg.write_text(make_svg(args.title, rows), encoding="utf-8")
    if args.summary_csv is not None:
        write_summary_csv(args.summary_csv, rows)
    print(f"Wrote: {args.svg}")
    if args.summary_csv is not None:
        print(f"Wrote: {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
