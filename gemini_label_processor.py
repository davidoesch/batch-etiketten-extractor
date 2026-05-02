import os
import sys
import time
import shutil
import json
import argparse
import re
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

def process_file(file_path: Path, output_dir: Path, client: genai.Client, current_idx: int, total_files: int):
    """Slices the label, sends it to Gemini, and generates JSON. Safely skips already processed files."""
    if file_path.suffix.lower() not in SUPPORTED_EXTS:
        return

    # Check if this file has already been processed successfully
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
        return

    # 2. STRICT ANCHOR-BASED GEMINI PROMPT
    prompt = """
    Look at this cropped and rotated image. We are looking for a specific white label with dot-matrix text.

    CRITICAL: If the image does NOT contain a white label with text (for example, if it is just a grey sleeve, a logo, or blank), you MUST return  string ("nodata") for ALL fields. DO NOT invent numbers or copy examples.

    If the label IS present, it has two distinct columns of text separated by a vertical divider line or a column of "¦" characters. The text is arranged in a strict 3-row grid, with the Left Column containing 3 lines of text and the Right Column containing 3 lines of text. The horizontal alignment of the text is CRITICAL to understand which field is which.

    *** GRID ALIGNMENT RULES ***
    The Left Column acts as your absolute horizontal ruler. You MUST match the text in the Right Column to the exact horizontal baselines of the Left Column:
    - `field1`: Left Column, Top line
    - `field2`: Left Column, Middle line
    - `field3`: Left Column, Bottom line

    - `field4`: Right Column text that is horizontally aligned with `field1`.
    - `field5`: Right Column text that is horizontally aligned with `field2`.
    - `field6`: Right Column text that is horizontally aligned with `field3`.

    CRITICAL ALIGNMENT CONSTRAINT: DO NOT let text float up. If there is text aligned with `field1` and text aligned with `field3`, but physical empty space aligned with `field2`, you MUST output `field5` as "" and put the bottom text in `field6`.

    *** SEPARATE CODES ***
    Separate from this 3-row grid, on the far Bottom Right of the label, sits the ID number and hyphenated code. DO NOT place these into field4, field5, or field6.
    - `id_number`: The 5 to 7 digit number (Bottom right, last numbers on the label).CRITICAL: we only deal with numbers 0-9, no letters in this field.
    - `hyphenated_code`: The hyphenated code (Bottom right, just before the ID Number).

    If a specific spot in the grid is physically empty, leave its value as string ("nodata"). Do not return conversational text, just the JSON.
    """

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

            # 3. MERGE FILENAME & ABORT IF MISSING DATA
            result["filename"] = file_path.stem

            id_num = result.get("id_number", "").strip()
            hyph_code = result.get("hyphenated_code", "").strip()

            if not id_num or not hyph_code:
                print(f"[Core fields missing: '{id_num}', '{hyph_code}'] -> Skipping JSON creation.", flush=True)
                with open(output_dir / "error_files.txt", "a") as f:
                    f.write(f"MISSING_DATA: {file_path.name}\n")
                return  # Instantly exit to prevent saving the empty JSON

            # 4. SAVE JSON
            expected_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"JSON Created -> {expected_json_path.name}")

            break # Success, exit retry loop

        except APIError as e:
            error_msg = str(e)
            if '429' in error_msg or '503' in error_msg:
                wait_time = 30 * (attempt + 1)

                if attempt == max_retries - 1:
                    print(f"\n[FATAL] Hit max retries for {file_path.name}. Likely hit Daily Quota.")
                    with open(output_dir / "error_files.txt", "a") as f:
                        f.write(f"API_QUOTA_FAILED: {file_path.name}\n")
                    break

                print(f"\n[Attempt {attempt+1}/{max_retries}] Server busy or Rate Limit. Pausing for {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"\nGemini API Error: {e}")
                with open(output_dir / "error_files.txt", "a") as f:
                    f.write(f"UNEXPECTED_API_ERROR: {file_path.name}\n")
                break
        except Exception as e:
            print(f"\nFailed to parse response: {e}")
            with open(output_dir / "error_files.txt", "a") as f:
                f.write(f"PARSE_ERROR: {file_path.name}\n")
            break

def main():
    parser = argparse.ArgumentParser(description="Consolidate label data and log failures locally without moving files.")
    parser.add_argument("input_dir", type=str, help="Folder containing incoming photos")
    parser.add_argument("output_dir", type=str, help="Folder where JSON metadata will be created")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    error_log_path = output_dir / "error_files.txt"
    if not error_log_path.exists():
        with open(error_log_path, "w") as f:
            f.write("# Error Log - Failed files\n")

    client = genai.Client()

    print("=== Bulk Label Metadata Consolidation ===")
    print(f"Input Images:   {input_dir}")
    print(f"Output Metadata: {output_dir}")
    print("Files will NOT be moved or copied.\n")

    files_to_process = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]

    # --- MODIFICATION B: Robust Natural Sorting ---
    # Ensures file_2 comes before file_10.
    files_to_process.sort(key=natural_sort_key)

    total_files = len(files_to_process)

    if not files_to_process:
        print("No supported images found in input directory.")
        return

    print(f"Found {total_files} files in directory. Starting process...\n")

    for idx, p in enumerate(files_to_process):
        # Pass idx + 1 and the total_files down to the process function
        process_file(p, output_dir, client, idx + 1, total_files)

        # Free Tier Rate Limit Management
        if idx < total_files - 1:
            time.sleep(4)

    print("\nBatch processing complete! Check the output directory for JSON files and 'error_files.txt'.")

if __name__ == "__main__":
    main()