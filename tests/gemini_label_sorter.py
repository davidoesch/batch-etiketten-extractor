import os
import sys
import time
import shutil
import json
import argparse
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

def process_file(file_path: Path, output_dir: Path, client: genai.Client, move: bool, save_ocr: bool):
    """Crops the label, sends it to Gemini, and sorts the original image."""
    if file_path.suffix.lower() not in SUPPORTED_EXTS:
        return

    print(f"Processing: {file_path.name}...", end=" ", flush=True)

    # 1. LOCAL PRIVACY CROP
    try:
        img = Image.open(file_path)
        w, h = img.size
        crop_box = (int(w * 0.70), 0, w, h)
        label_crop = img.crop(crop_box)
        label_crop = label_crop.transpose(Image.ROTATE_90)
    except Exception as e:
        print(f"Failed to read/crop image locally: {e}")
        return

    # 2. SEND TO GEMINI
    prompt = """
    Look at this cropped label. Please extract two specific pieces of information:
    1. The 5 to 7 digit ID number (for example: 108480, 110895, 108477).
    2. The hyphenated code (for example: 2-OR-89, 2-OR-90, 1-DR-90).

    Return ONLY a valid JSON object with the keys 'id_number' and 'hyphenated_code'.
    If you cannot find one of them, leave the value as an empty string.
    """

    # Increased to 5 retries to handle stubborn 503 server spikes
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

            folder_name = result.get("id_number", "").strip()
            if not folder_name:
                folder_name = "UNSORTED"

            # 3. SORT AND MOVE
            dest_folder = output_dir / folder_name
            dest_folder.mkdir(parents=True, exist_ok=True)
            dest_file = dest_folder / file_path.name

            if move:
                shutil.move(str(file_path), str(dest_file))
                print(f"Moved -> {folder_name}/")
            else:
                shutil.copy2(str(file_path), str(dest_file))
                print(f"Copied -> {folder_name}/")

            if save_ocr:
                json_path = dest_folder / f"{file_path.stem}.ocr.json"
                json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

            break # Success, exit retry loop

        except APIError as e:
            # Check if the error is a 429 (Rate Limit) or 503 (Unavailable)
            error_msg = str(e)
            if '429' in error_msg or '503' in error_msg:
                wait_time = 30 # Wait 30 seconds to let the server cool down
                print(f"\n[Attempt {attempt+1}/{max_retries}] Server busy (503) or Rate Limit (429). Pausing for {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"\nGemini API Error: {e}")
                break
        except Exception as e:
            print(f"\nFailed to parse response: {e}")
            break

def main():
    parser = argparse.ArgumentParser(description="Crop labels for privacy, extract text via Gemini, and sort.")
    parser.add_argument("input_dir", type=str, help="Folder containing incoming photos")
    parser.add_argument("output_dir", type=str, help="Folder where sorted photos will go")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument("--save-ocr", action="store_true", help="Save a .json file with the extracted data")

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
    client = genai.Client()

    print("=== Privacy-First Gemini Label Sorter ===")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    files_to_process = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]

    if not files_to_process:
        print("No supported images found in input directory.")
        return

    print(f"Found {len(files_to_process)} files to process. Starting batch...\n")

    for idx, p in enumerate(files_to_process):
        process_file(p, output_dir, client, args.move, args.save_ocr)

        # Free Tier Rate Limit Management (~15 Requests per Minute)
        if idx < len(files_to_process) - 1:
            time.sleep(4)

    print("\nBatch processing complete!")

if __name__ == "__main__":
    main()