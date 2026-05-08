#!/usr/bin/env python3

import csv
from pathlib import Path

# Input CSV
csv_file = Path("/home/menas/Downloads/nade/0001-7023_results.csv")

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

    # Replace single invalid values
    if value in ["", "I", "1", "|", "¦", ";"]:
        return "nodata"

    return value


with open(csv_file, "r", encoding="utf-8", newline="") as infile, \
     open(output_file, "w", encoding="utf-8", newline="") as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        cleaned_row = [clean_value(cell) for cell in row]
        writer.writerow(cleaned_row)

print(f"Cleaned CSV written to: {output_file}")