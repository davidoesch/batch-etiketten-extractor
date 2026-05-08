import os
import sys
import time
import json
import argparse
import re
import csv
from pathlib import Path

# --- Dependencies ---
try:
    from PIL import Image
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
except ImportError:
    print("Error: Missing required packages. Please run:")
    print("pip install pillow google-genai")
    sys.exit(1)

# --- Configuration ---
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

def natural_sort_key(path: Path):
    """Sorts files naturally so 'img_2' comes before 'img_10'."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path.name)]

def write_failed_json(file_path: Path, output_dir: Path):
    """Handle catastrophic errors by generating a JSON filled with 'nodata'."""
    result = {
        "filename": file_path.stem,
        "id_number": "nodata",
        "hyphenated_code": "nodata",
        "field1": "nodata",
        "field2": "nodata",
        "field3": "nodata",
        "field4": "nodata",
        "field5": "nodata",
        "field6": "nodata"
    }
    expected_json_path = output_dir / f"{file_path.stem}.json"
    expected_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"JSON Created (error fallback) -> {expected_json_path.name}")

def process_file(file_path: Path, output_dir: Path, client: genai.Client, current_idx: int, total_files: int, config: dict):
    """Slices the label, sends it to Gemini, and generates JSON. Safely skips already processed files."""
    if file_path.suffix.lower() not in SUPPORTED_EXTS:
        return

    expected_json_path = output_dir / f"{file_path.stem}.json"
    if expected_json_path.exists():
        print(f"Skipping {current_idx} of {total_files}: {file_path.name} (JSON already exists)")
        return

    print(f"Processing {current_idx} of {total_files}: {file_path.name}...", end=" ", flush=True)

    # 1. LOCAL PRIVACY CROP & ROTATION
    try:
        img = Image.open(file_path)
        w, h = img.size
        crop_box = (int(w * 0.70), 0, w, h)
        label_crop = img.crop(crop_box)
        label_crop = label_crop.transpose(Image.ROTATE_90)
    except Exception as e:
        print(f"\nFailed to read/crop image locally: {e}")
        write_failed_json(file_path, output_dir)
        return

    # 2. STRICT ANCHOR-BASED GEMINI PROMPT
    prompt = """
    Look at this cropped and rotated image. We are looking for a specific white label with dot-matrix text.

    CRITICAL: If the image does NOT contain a white label with text (for example, if it is just a grey sleeve, a logo, or blank), you MUST return "nodata" for ALL fields. DO NOT invent numbers or copy examples.

    If the label IS present, it has two distinct columns of text separated by a vertical divider line or a column of "¦" characters. The text is arranged in a strict 3-row grid, with the Left Column containing 3 lines of text and the Right Column containing 3 lines of text. The horizontal alignment of the text is CRITICAL to understand which field is which.

    *** GRID ALIGNMENT RULES ***
    The Left Column acts as your absolute horizontal ruler. You MUST match the text in the Right Column to the exact horizontal baselines of the Left Column:
    - `field1`: Left Column, Top line
    - `field2`: Left Column, Middle line
    - `field3`: Left Column, Bottom line

    - `field4`: Right Column text that is horizontally aligned with `field1`.
    - `field5`: Right Column text that is horizontally aligned with `field2`.
    - `field6`: Right Column text that is horizontally aligned with `field3`.

    CRITICAL ALIGNMENT CONSTRAINT: DO NOT let text float up. If there is text aligned with `field1` and text aligned with `field3`, but physical empty space aligned with `field2`, you MUST output `field5` as "nodata" and put the bottom text in `field6`.

    *** SEPARATE CODES ***
    Separate from this 3-row grid, on the far Bottom Right of the label, sits the ID number and hyphenated code. DO NOT place these into field4, field5, or field6.
    - `id_number`: The 5 to 7 digit number (Bottom right, last numbers on the label).CRITICAL: we only deal with numbers 0-9, no letters in this field.
    - `hyphenated_code`: The hyphenated code (Bottom right, just before the ID Number).

    If a specific spot in the grid is physically empty, leave its value as "nodata". Do not return conversational text, just the JSON.
    """

    # --- The 429/503 "Take a nap" Retry Logic ---
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[label_crop, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )

            result = json.loads(response.text)
            result["filename"] = file_path.stem

            expected_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"JSON Created -> {expected_json_path.name}")
            break # Success, exit retry loop

        except APIError as e:
            error_msg = str(e)
            if '429' in error_msg or '503' in error_msg:

                # --- AUTO-DETECT FREE TIER ---
                if '429' in error_msg and config["is_fast_mode"]:
                    print("\n[!] 429 Rate Limit Hit: You are currently using a Free Tier API Key.")
                    print("[!] Auto-adjusting to Free Tier Mode (adding 4-second delays between files).")
                    config["is_fast_mode"] = False # Tells the main loop to start pausing

                wait_time = 30 * (attempt + 1)

                if attempt == max_retries - 1:
                    print(f"\n[FATAL] Hit max retries for {file_path.name}. API is completely unresponsive.")
                    write_failed_json(file_path, output_dir)
                    break

                print(f"\n[Attempt {attempt+1}/{max_retries}] Taking a nap for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print(f"\nGemini API Error: {e}")
                write_failed_json(file_path, output_dir)
                break
        except Exception as e:
            print(f"\nFailed to parse response: {e}")
            write_failed_json(file_path, output_dir)
            break

def generate_csv_from_jsons(output_dir: Path, output_csv: Path):
    """Convert generated JSON files into a consolidated CSV natively."""
    fieldnames = [
        "filename", "id_number", "hyphenated_code",
        "field1", "field2", "field3",
        "field4", "field5", "field6"
    ]

    json_files = list(output_dir.glob("*.json"))
    if not json_files:
        print("\nNo JSON files found in the output directory to compile into CSV.")
        return

    print(f"\nMerging {len(json_files)} JSON files into CSV...")
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        success_count = 0
        for json_path in json_files:
            try:
                with open(json_path, mode="r", encoding="utf-8") as jf:
                    data = json.load(jf)
                # Use "nodata" as a fallback if the key is missing entirely
                row = {fn: data.get(fn, "nodata") for fn in fieldnames}
                writer.writerow(row)
                success_count += 1
            except Exception as e:
                print(f"Error processing {json_path.name}: {e}")

    print(f"Done! Successfully generated final CSV:")
    print(f"-> {output_csv.absolute()}")

def main():
    parser = argparse.ArgumentParser(description="Consolidate label data directly to JSON and CSV.")
    parser.add_argument("input_dir", type=str, help="Folder containing incoming photos")

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist or is not a folder.")
        sys.exit(1)

    output_dir = input_dir.parent / f"{input_dir.name}_metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = input_dir.parent / f"{input_dir.name}_result.csv"

    print("=== Bulk Label Metadata Consolidation ===\n")

    # Strict Key Check and Instructions
    key_file = Path(__file__).parent / "key.txt"
    if key_file.exists():
        api_key = key_file.read_text(encoding="utf-8").strip()
    else:
        api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key or api_key == "testing key":
        print("[!] CRITICAL ERROR: No valid GEMINI_API_KEY found.")
        print("Google's Gemini API strictly requires a real API key to function, even on the Free Tier.")
        print("\nHow to fix this:")
        print("1. Get your free key from: https://aistudio.google.com/app/apikey")
        print("2. Create a file called key.txt in the same folder as this script.")
        print("3. Paste your API key as the only content of that file and save it.")
        print("4. Run this python script again.")
        sys.exit(1)

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Gemini Client: {e}")
        sys.exit(1)

    # We assume Fast Mode initially. If we hit a 429 error, the script will dynamically change this to False.
    config = {"is_fast_mode": True}

    print("Mode: Auto-Detect (Starting in Fast Mode)")
    print(f"Input Images:    {input_dir}")
    print(f"Output Metadata: {output_dir}")
    print(f"Output CSV:      {output_csv}\n")

    files_to_process = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    files_to_process.sort(key=natural_sort_key)
    total_files = len(files_to_process)

    if not files_to_process:
        print("No supported images found in input directory.")
        return

    print(f"Found {total_files} files in directory. Starting process...\n")

    for idx, p in enumerate(files_to_process):
        process_file(p, output_dir, client, idx + 1, total_files, config)

        # Rate Limit Management: Only pauses if the script automatically downgraded to Free Tier mode
        if not config["is_fast_mode"] and idx < total_files - 1:
            time.sleep(4)

    print("\nBatch JSON processing complete!")

    generate_csv_from_jsons(output_dir, output_csv)

if __name__ == "__main__":
    main()