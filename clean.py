#!/usr/bin/env python3

import csv
from pathlib import Path

# Input CSV
csv_file = Path("/media/menas/data/projects/nade/0001-7023_result.csv")

# Output CSV
output_file = csv_file.with_name(csv_file.stem + "_cleaned.csv")


def clean_value(value):
    """
    Cleans a CSV field:
    - removes "|" characters
    - removes leading ":" characters
    - removes leading "I " patterns
    - removes leading/trailing whitespace
    - replaces empty fields with "nodata"
    """

    if value is None:
        return "nodata"

    # Remove all pipe characters
    value = value.replace("|", "")

    # Remove leading/trailing whitespace
    value = value.strip()

    # Remove leading ":" characters
    while value.startswith(":"):
        value = value[1:].lstrip()

    # Remove leading ":" characters
    while value.startswith("¦"):
        value = value[1:].lstrip()

    # Remove leading ":" characters
    while value.startswith(";"):
        value = value[1:].lstrip()

    # Remove leading "I "
    while value.startswith("I "):
        value = value[2:].lstrip()

    # Remove leading "! "
    while value.startswith("! "):
        value = value[2:].lstrip()

    # Replace single invalid values
    if value in ["", "I", "1", "|", "¦", ";"]:
        return "nodata"

    return value


def clean_row(row):
    """Apply cross-field cleaning rules to a cleaned row dict."""
    id_num = row.get("id_number", "")
    hyph = row.get("hyphenated_code", "")
    field6 = row.get("field6", "")

    # A: Remove id_number exact string from field6
    if id_num and id_num != "nodata" and id_num in field6:
        field6 = field6.replace(id_num, "").strip()

    # B: Remove hyphenated_code exact string from field6
    if hyph and hyph != "nodata" and hyph in field6:
        field6 = field6.replace(hyph, "").strip()

    row["field6"] = field6 if field6 else "nodata"

    # C: Replace B with 8 in id_number (OCR confusion)
    # D: Replace Z with 2 in id_number (OCR confusion)
    if id_num and id_num != "nodata":
        row["id_number"] = id_num.replace("B", "8").replace("Z", "2")

    return row


with open(csv_file, "r", encoding="utf-8", newline="") as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames

    rows = []
    for row in reader:
        cleaned = {k: clean_value(v) for k, v in row.items()}
        cleaned = clean_row(cleaned)
        rows.append(cleaned)

# E: Sort by filename treated as integer
rows.sort(key=lambda r: int(r.get("filename", 0)) if str(r.get("filename", "")).isdigit() else 0)

with open(output_file, "w", encoding="utf-8", newline="") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Cleaned CSV written to: {output_file}")
