import json
import csv
import argparse
from pathlib import Path

def generate_csv_from_jsons(input_dir: Path, output_csv: Path):
    # The exact columns we want in our CSV, in the requested order
    fieldnames = [
        "filename",
        "id_number",
        "hyphenated_code",
        "field1",
        "field2",
        "field3",
        "field4",
        "field5",
        "field6"
    ]

    # Find all JSON files in the directory
    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON files. Creating CSV...")

    # Open the CSV file for writing (using utf-8 encoding for special characters like ÄÖÜẞ)
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        success_count = 0

        # Read each JSON file and write its contents to the CSV
        for json_path in json_files:
            try:
                with open(json_path, mode="r", encoding="utf-8") as jf:
                    data = json.load(jf)

                # Extract only the specific fields we care about (ignoring any extra stuff if it exists)
                # We use .get() so if a field is missing entirely, it just defaults to an empty string
                row = {
                    "filename": data.get("filename", ""),
                    "id_number": data.get("id_number", ""),
                    "hyphenated_code": data.get("hyphenated_code", ""),
                    "field1": data.get("field1", ""),
                    "field2": data.get("field2", ""),
                    "field3": data.get("field3", ""),
                    "field4": data.get("field4", ""),
                    "field5": data.get("field5", ""),
                    "field6": data.get("field6", "")
                }

                writer.writerow(row)
                success_count += 1

            except json.JSONDecodeError:
                print(f"Warning: Could not parse {json_path.name}. It might be corrupted.")
            except Exception as e:
                print(f"Error processing {json_path.name}: {e}")

    print(f"\nDone! Successfully merged {success_count} JSON files into:")
    print(f"-> {output_csv.absolute()}")

def main():
    parser = argparse.ArgumentParser(description="Convert a folder of label JSONs into a single CSV file.")
    parser.add_argument("input_dir", type=str, help="Folder containing the extracted JSON files")
    parser.add_argument("output_csv", type=str, help="Path and filename for the output CSV (e.g., output/results.csv)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Directory '{input_dir}' does not exist.")
        return

    # Create the output directory if it doesn't exist
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    generate_csv_from_jsons(input_dir, output_csv)

if __name__ == "__main__":
    main()