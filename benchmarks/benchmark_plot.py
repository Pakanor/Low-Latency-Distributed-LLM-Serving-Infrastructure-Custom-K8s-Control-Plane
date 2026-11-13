#!/usr/bin/env python3
"""Create benchmark charts from saved CSV results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def write_throughput_svg(csv_path: Path, output_path: Path) -> None:
    rows = list(csv.DictReader(csv_path.open(newline="")))
    if not rows:
        raise ValueError(f"No benchmark rows found in {csv_path}")
    required = {"variant", "max_parallel", "tokens_per_second"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Missing columns: {sorted(required - set(rows[0]))}")

    width, height = 760, 420
    chart_left, chart_top, chart_width, chart_height = 80, 40, 640, 280
    max_value = max(float(row["max_parallel"]) for row in rows) or 1.0
    bar_width = min(180, chart_width // max(len(rows), 1) - 20)
    gap = 20
    colors = {"paged": "#176b87", "contiguous": "#d97745"}
    bars = []
    for index, row in enumerate(rows):
        value = float(row["max_parallel"])
        bar_height = chart_height * value / max_value
        x = chart_left + 40 + index * (bar_width + gap)
        y = chart_top + chart_height - bar_height
        variant = row["variant"]
        bars.append(
            f'<rect x="{x}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" '
            f'fill="{colors.get(variant, "#52616b")}" rx="4" />'
            f'<text x="{x + bar_width / 2}" y="{y - 10:.1f}" text-anchor="middle">'
            f'{value:g}</text>'
            f'<text x="{x + bar_width / 2}" y="{chart_top + chart_height + 28}" '
            f'text-anchor="middle">{variant}</text>'
        )
    output_path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <style>text {{ font: 16px sans-serif; fill: #25323a; }} .title {{ font-size: 20px; font-weight: 600; }}</style>
  <rect width="100%" height="100%" fill="#f7f4ee" />
  <text x="{width / 2}" y="25" text-anchor="middle" class="title">Maximum concurrent sequences</text>
  <line x1="{chart_left}" y1="{chart_top + chart_height}" x2="{chart_left + chart_width}" y2="{chart_top + chart_height}" stroke="#25323a" />
  <line x1="{chart_left}" y1="{chart_top}" x2="{chart_left}" y2="{chart_top + chart_height}" stroke="#25323a" />
  {''.join(bars)}
</svg>
'''
    )


def main() -> None:
    args = build_parser().parse_args()
    csv_path = args.results_dir / "throughput_results.csv"
    output_path = args.output or args.results_dir / "throughput.svg"
    write_throughput_svg(csv_path, output_path)


if __name__ == "__main__":
    main()
