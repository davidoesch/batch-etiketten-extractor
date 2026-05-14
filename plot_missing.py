#!/usr/bin/env python3
"""Visualize gaps in the 'filename' column of a CSV.

Reads zero-padded integers from the 'filename' column, identifies missing
numbers in the range [1, MAX], groups consecutive misses into gap ranges,
and writes a multi-panel diagnostic PNG.
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_present_numbers(csv_path: Path) -> set[int]:
    """Return the set of integers present in the 'filename' column."""
    present: set[int] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "filename" not in (reader.fieldnames or []):
            raise ValueError(f"Column 'filename' not found. Header: {reader.fieldnames}")
        for row in reader:
            raw = (row.get("filename") or "").strip()
            if not raw:
                continue
            try:
                present.add(int(raw))
            except ValueError:
                continue
    return present


def group_gaps(missing: list[int]) -> list[tuple[int, int]]:
    """Group a sorted list of missing numbers into inclusive (start, end) ranges."""
    if not missing:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = missing[0]
    for n in missing[1:]:
        if n == prev + 1:
            prev = n
        else:
            groups.append((start, prev))
            start = prev = n
    groups.append((start, prev))
    return groups


def plot_gaps(
    groups: list[tuple[int, int]],
    upper: int,
    n_present: int,
    out_path: Path,
    top_n: int = 20,
) -> None:
    """Render a four-panel diagnostic figure of missing-number gaps."""
    n_missing = sum(end - start + 1 for start, end in groups)
    gap_sizes = [end - start + 1 for start, end in groups]

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 2.5], hspace=0.55, wspace=0.25)

    # Panel 1: presence strip (green = present, red = missing)
    ax_strip = fig.add_subplot(gs[0, :])
    ax_strip.set_xlim(0, upper + 1)
    ax_strip.set_ylim(0, 1)
    ax_strip.add_patch(patches.Rectangle(
        (1, 0), upper, 1, facecolor="#4caf50", edgecolor="none"
    ))
    for start, end in groups:
        ax_strip.add_patch(patches.Rectangle(
            (start, 0), end - start + 1, 1, facecolor="#e53935", edgecolor="none"
        ))
    ax_strip.set_yticks([])
    ax_strip.set_xlabel("ID number")
    ax_strip.set_title(
        f"Presence map: {n_present} present (green), "
        f"{n_missing} missing in {len(groups)} gap(s) (red)"
    )

    # Panel 2: gap-size histogram
    ax_hist = fig.add_subplot(gs[1, 0])
    if gap_sizes:
        ax_hist.hist(gap_sizes, bins=30, color="#e53935", edgecolor="black")
        if max(gap_sizes) > 10:
            ax_hist.set_yscale("log")
    ax_hist.set_xlabel("Gap size (consecutive missing numbers)")
    ax_hist.set_ylabel("Number of gaps")
    ax_hist.set_title("Gap-size distribution")

    # Panel 3: gap size vs. position
    ax_pos = fig.add_subplot(gs[1, 1])
    if groups:
        starts = [s for s, _ in groups]
        ax_pos.scatter(starts, gap_sizes, c="#e53935", s=25, alpha=0.6, edgecolors="black", linewidths=0.3)
        if max(gap_sizes) > 10:
            ax_pos.set_yscale("log")
    ax_pos.set_xlabel("Gap start (ID number)")
    ax_pos.set_ylabel("Gap size")
    ax_pos.set_title("Gap size vs. position (clusters indicate problem zones)")
    ax_pos.grid(True, alpha=0.3)

    # Panel 4: top N largest gaps
    ax_top = fig.add_subplot(gs[2, :])
    top = sorted(groups, key=lambda g: g[1] - g[0], reverse=True)[:top_n]
    if top:
        labels = [f"{s}-{e}" if s != e else f"{s}" for s, e in top]
        sizes = [e - s + 1 for s, e in top]
        y = list(range(len(top)))
        ax_top.barh(y, sizes, color="#e53935", edgecolor="black")
        ax_top.set_yticks(y)
        ax_top.set_yticklabels(labels, fontsize=9)
        ax_top.invert_yaxis()
        for i, size in enumerate(sizes):
            ax_top.text(size, i, f"  {size}", va="center", fontsize=9)
    ax_top.set_xlabel("Gap size (missing numbers)")
    ax_top.set_title(f"Top {min(top_n, len(groups))} largest gaps (ranges to investigate)")

    fig.suptitle(
        f"Missing-data analysis: range 1 to {upper}",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {out_path}")


def write_gap_report(groups: list[tuple[int, int]], out_path: Path) -> None:
    """Write a CSV listing every gap with start, end, and size."""
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gap_start", "gap_end", "gap_size"])
        for start, end in groups:
            writer.writerow([start, end, end - start + 1])
    print(f"Gap list saved to: {out_path}")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python plot_missing.py <csv_path> [expected_max] [--out <png>]")
        return 1

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        return 1

    expected_max: int | None = None
    out_png = Path("missing_gaps.png")
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--out" and i + 1 < len(args):
            out_png = Path(args[i + 1])
            i += 2
        else:
            expected_max = int(args[i])
            i += 1

    present = load_present_numbers(csv_path)
    if not present:
        print("No numbers parsed from CSV.")
        return 1

    upper = expected_max if expected_max is not None else max(present)
    missing = sorted(set(range(1, upper + 1)) - present)
    groups = group_gaps(missing)

    print(f"Present : {len(present)}")
    print(f"Range   : 1 to {upper}")
    print(f"Missing : {len(missing)} in {len(groups)} gap(s)")
    if groups:
        largest = max(groups, key=lambda g: g[1] - g[0])
        size = largest[1] - largest[0] + 1
        print(f"Largest : {largest[0]}-{largest[1]} ({size} numbers)")

    plot_gaps(groups, upper, len(present), out_png)
    write_gap_report(groups, out_png.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())