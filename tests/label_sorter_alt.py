import os
import sys
import time
import shutil
import argparse
import re
import json
from pathlib import Path

# --- Dependencies ---
try:
    from PIL import Image, ImageOps, ImageFilter
    import pytesseract
except ImportError:
    print("Error: Missing Pillow or pytesseract. Install with: pip install pillow pytesseract")
    sys.exit(1)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

# --- Configuration ---
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Default patterns prioritized for your specific dot-matrix labels
DEFAULT_PATTERNS = [
    r"\b(\d{5,7})\b",                                      # e.g., 110895
    r"\b([A-Z][0-9]{3,4})\b",                              # e.g., D681
    r"\b([0-9A-Z]{1,3}-[0-9A-Z]{2}-[0-9A-Z]{2,4})\b",      # e.g., 1-DR-90
    r"\b(CG\s+[A-ZÄÖÜẞ]+(?:\s+[A-ZÄÖÜẞ]+){1,4})\b",        # e.g., CG PAKISTAN KARACHI
]

def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """
    Enhance dot-matrix/low-contrast labels.
    Upscales the image and connects the dots for solid text reading.
    """
    try:
        import cv2
        import numpy as np
        arr = np.array(img)

        # Convert to grayscale
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # 1. UPSCALE: Make the image 2x larger so text is easier to read
        arr = cv2.resize(arr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # 2. BOOST CONTRAST
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        arr = clahe.apply(arr)

        # 3. BINARIZE: Force pure black text on pure white background
        thr = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 41, 15)

        # 4. CONNECT THE DOTS: Expand the black pixels slightly to fill dot-matrix gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thr = cv2.erode(thr, kernel, iterations=1)

        # 5. DESKEW: Straighten the text
        coords = np.column_stack(np.where(thr == 0))
        angle = 0.0
        if coords.size:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

        (h, w) = thr.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        deskew = cv2.warpAffine(thr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return Image.fromarray(deskew)

    except ImportError:
        # Pillow fallback if OpenCV is not installed
        g = ImageOps.grayscale(img)
        g = ImageOps.autocontrast(g, cutoff=1)
        g = g.resize((int(g.width * 2.0), int(g.height * 2.0)))
        g = g.point(lambda x: 255 if x > 180 else 0)
        return g

def ocr_image(path: Path) -> str:
    """Reads the image, crops out the noisy film slide, and extracts text."""
    img = Image.open(path)

    # 1. Rotate counter-clockwise to align text horizontally.
    # The label (originally on the right) is now at the TOP of the image.
    img = img.transpose(Image.ROTATE_90)

    # 2. CROP: Keep only the top 30% of the image where the label is.
    # This throws away the dark slide and plastic glare.
    width, height = img.size
    crop_box = (0, 0, width, int(height * 0.30)) # left, upper, right, lower
    img = img.crop(crop_box)

    try:
        img = preprocess_for_ocr(img)
        # --psm 11 looks for sparse text; whitelist prevents letter/number confusion
        cfg = "--oem 1 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/. "
        return pytesseract.image_to_string(img, config=cfg)
    finally:
        img.close()

def find_label(text: str, patterns: list) -> tuple:
    """Matches OCR text against regex patterns to find the folder name."""
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            # Return the first captured group
            return match.group(1).strip(), pattern
    return "UNSORTED", None

def process_file(file_path: Path, output_dir: Path, patterns: list, move: bool, save_ocr: bool):
    """Handles the OCR, sorting, and moving/copying of a single image."""
    if file_path.suffix.lower() not in SUPPORTED_EXTS:
        return

    print(f"\nProcessing: {file_path.name}")
    try:
        ocr_text = ocr_image(file_path)
        label, matched_pattern = find_label(ocr_text, patterns)

        # Create destination folder
        dest_folder = output_dir / label
        dest_folder.mkdir(parents=True, exist_ok=True)

        dest_file = dest_folder / file_path.name

        # Move or copy the file
        if move:
            shutil.move(str(file_path), str(dest_file))
            print(f"Moved to -> {dest_folder.name}/")
        else:
            shutil.copy2(str(file_path), str(dest_file))
            print(f"Copied to -> {dest_folder.name}/")

        # Save OCR sidecar files
        if save_ocr:
            txt_path = dest_folder / f"{file_path.stem}.ocr.txt"
            txt_path.write_text(ocr_text, encoding="utf-8")

            json_path = dest_folder / f"{file_path.stem}.ocr.json"
            json_data = {
                "label": label,
                "matched_pattern": matched_pattern,
                "raw_text": ocr_text.strip()
            }
            json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")

    except Exception as e:
        print(f"Failed to process {file_path.name}: {e}")

class ImageHandler(FileSystemEventHandler):
    def __init__(self, output_dir, patterns, move, save_ocr):
        self.output_dir = output_dir
        self.patterns = patterns
        self.move = move
        self.save_ocr = save_ocr

    def on_created(self, event):
        if not event.is_directory:
            # Wait a brief moment to ensure the file is fully written by the OS
            time.sleep(1)
            process_file(Path(event.src_path), self.output_dir, self.patterns, self.move, self.save_ocr)

def main():
    parser = argparse.ArgumentParser(description="Read label text from JPGs and auto-sort them into folders.")
    parser.add_argument("input_dir", type=str, help="Folder containing incoming photos")
    parser.add_argument("output_dir", type=str, help="Folder where sorted photos will be moved")
    parser.add_argument("--watch", action="store_true", help="Continuously watch the input folder")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument("--save-ocr", action="store_true", help="Save .txt and .json files with the OCR text")
    parser.add_argument("--pattern", action="append", help="Regex pattern to extract label (can be used multiple times)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = args.pattern if args.pattern else DEFAULT_PATTERNS

    print("=== Label Sorter ===")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    # Process existing files first
    existing_files = [p for p in input_dir.iterdir() if p.is_file()]
    if existing_files:
        print(f"Found {len(existing_files)} existing files. Processing...")
        for p in existing_files:
            process_file(p, output_dir, patterns, args.move, args.save_ocr)

    # Start watch mode if requested
    if args.watch:
        if not WATCHDOG_AVAILABLE:
            print("Error: Watch mode requires the 'watchdog' package. (pip install watchdog)")
            sys.exit(1)

        print(f"\nWatching {input_dir} for new images... (Press Ctrl+C to stop)")
        event_handler = ImageHandler(output_dir, patterns, args.move, args.save_ocr)
        observer = Observer()
        observer.schedule(event_handler, str(input_dir), recursive=False)
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            print("\nStopped watching.")
        observer.join()

if __name__ == "__main__":
    main()