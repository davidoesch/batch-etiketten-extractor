#!/usr/bin/env python3
"""Find missing numbers in the 'filename' column of a CSV.

The 'filename' column contains zero-padded integers (e.g. '0001', '00010', '000100').
Identifies which integers in the range [1, MAX] are not present in the file.
"""

import csv
import sys
from pathlib import Path


def find_missing_numbers(
    csv_path: Path,
    expected_max: int | None = None,
) -> tuple[list[int], int, int]:
    """Read the CSV and return missing numbers plus summary stats.

    Args:
        csv_path: Path to the input CSV.
        expected_max: Upper bound of the expected range. If None, the maximum
            value found in the file is used.

    Returns:
        Tuple of (sorted missing numbers, count present, upper bound used).
    """
    present: set[int] = set()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "filename" not in (reader.fieldnames or []):
            raise ValueError(
                f"Column 'filename' not found. Header: {reader.fieldnames}"
            )
        for row_idx, row in enumerate(reader, start=2):
            raw = (row["filename"] or "").strip()
            if not raw:
                continue
            try:
                present.add(int(raw))
            except ValueError:
                print(
                    f"Warning: row {row_idx}: cannot parse '{raw}' as int",
                    file=sys.stderr,
                )

    if not present:
        return [], 0, 0

    upper = expected_max if expected_max is not None else max(present)
    full_range = set(range(1, upper + 1))
    missing = sorted(full_range - present)
    return missing, len(present), upper


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: python find_missing.py <csv_path> [expected_max] [--out <file>]"
        )
        return 1

    csv_path = Path(sys.argv[1])
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        return 1

    expected_max: int | None = None
    out_path: Path | None = None
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--out" and i + 1 < len(args):
            out_path = Path(args[i + 1])
            i += 2
        else:
            expected_max = int(args[i])
            i += 1

    missing, n_present, upper = find_missing_numbers(csv_path, expected_max)

    print(f"Present entries : {n_present}")
    print(f"Range checked   : 1 to {upper}")
    print(f"Missing count   : {len(missing)}")

    if out_path is not None:
        out_path.write_text("\n".join(str(n) for n in missing) + "\n", encoding="utf-8")
        print(f"Missing list written to: {out_path}")
    elif missing:
        print("Missing numbers:")
        for n in missing:
            print(n)

    return 0


if __name__ == "__main__":
    sys.exit(main())