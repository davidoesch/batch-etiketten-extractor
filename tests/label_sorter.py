#!/usr/bin/env python3
"""
label_sorter.py
---------------
Watches a folder for new images, reads label text via OCR (Tesseract),
extracts a key value, and moves/copies the image into a matching subfolder.

Supports: JPG, PNG, TIFF, BMP, WEBP  +  RAW (CR2, NEF, ARW, DNG, …) via rawpy

Usage:
    python label_sorter.py INPUT OUTPUT [--watch] [--move] [--save-ocr]
                            [--pattern REGEX] [--lang LANG]

Examples:
    # Copy (default), no watch, German OCR
    python label_sorter.py ./in ./out --lang deu

    # Watch + move + save OCR sidecars, custom pattern
    python label_sorter.py ./in ./out --watch --move --save-ocr \
        --pattern "Part\s*No\.?\s*[:#-]?\s*([A-Za-z0-9._-]+)"

    # macOS – optimised for dot-matrix labels like the sample
    python3 label_sorter.py Incoming Sorted --watch --move --save-ocr
"""

import argparse
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional: hard-code Tesseract path here if it is not in your system PATH
# Windows example:
#   import pytesseract
#   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# macOS / Linux: usually not needed after `brew install tesseract`
# ---------------------------------------------------------------------------

try:
    from PIL import Image, ImageOps, ImageFilter
    import numpy as np
except ImportError as exc:
    sys.exit(
        "Error: Pillow and numpy are required.\n"
        "Install: pip install Pillow numpy\n"
        f"Details: {exc}"
    )

try:
    import pytesseract
except ImportError:
    sys.exit(
        "Error: pytesseract is required.\n"
        "Install: pip install pytesseract"
    )

# RAW support (optional)
RAWPY_AVAILABLE = False
try:
    import rawpy  # type: ignore
    RAWPY_AVAILABLE = True
except ImportError:
    pass

# OpenCV support (optional but recommended for dot-matrix labels)
CV2_AVAILABLE = False
try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
    print("OpenCV is available: using it for preprocessing.")
except ImportError:
    pass

# watchdog (optional, needed only for --watch)
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
RAW_EXTS = {
    ".cr2", ".cr3", ".nef", ".arw", ".raf", ".dng",
    ".orf", ".rw2", ".srw", ".pef", ".nrw", ".kdc", ".sr2",
}
UNSORTED_DIR = "UNSORTED"

# Default extraction patterns – first match wins.
# Tuned for labels like the sample (D681 / 1-DR-90 / 110895).
DEFAULT_PATTERNS = [
    r"\b(\d{5,7})\b",                                          # 5–7 digit id  e.g. 110895
    r"\b([A-Z][0-9]{3,4})\b",                                  # letter+digits e.g. D681
    r"\b([0-9A-Z]{1,3}-[0-9A-Z]{2}-[0-9A-Z]{2,4})\b",         # hyphenated    e.g. 1-DR-90
    r"(?:Part\s*No\.?|P/N|PN)\s*[:#-]?\s*([A-Za-z0-9._-]{3,})",
    r"(?:Serial(?:\s*(?:No\.?|#))?|S/?N)\s*[:#-]?\s*([A-Za-z0-9._-]{4,})",
    r"(?:Model)\s*[:#-]?\s*([A-Za-z0-9._-]{3,})",
    r"(?:Batch|Lot)\s*[:#-]?\s*([A-Za-z0-9._-]{3,})",
    r"(?:SKU)\s*[:#-]?\s*([A-Za-z0-9._-]{3,})",
    r"\b([A-Z0-9]{4,10})\b",                                   # generic ALLCAPS token fallback
]

# Tesseract OCR config – whitelist keeps only label-relevant characters
TESS_CONFIG = (
    "--oem 1 --psm 6 "
    "-c tessedit_char_whitelist="
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "0123456789-/. "
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def is_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTS | RAW_EXTS


def _open_raw_as_pil(path: Path) -> Image.Image:
    if not RAWPY_AVAILABLE:
        raise RuntimeError(
            f"RAW file '{path.name}' requires rawpy.\n"
            "Install: pip install rawpy"
        )
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(no_auto_bright=True, output_bps=8)
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 65535) / 257).astype(np.uint8)
    return Image.fromarray(rgb)


def load_image(path: Path) -> Image.Image:
    if path.suffix.lower() in RAW_EXTS:
        return _open_raw_as_pil(path)
    return Image.open(path)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """
    Enhance dot-matrix / low-contrast labels.
    Uses OpenCV when available; falls back to a Pillow pipeline.
    """
    if CV2_AVAILABLE:
        try:
            arr = np.array(img)
            if arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            # CLAHE – local contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            arr = clahe.apply(arr)
            # Bilateral filter preserves dot edges
            arr = cv2.bilateralFilter(arr, d=7, sigmaColor=15, sigmaSpace=15)
            # Adaptive threshold – binarise
            thr = cv2.adaptiveThreshold(
                arr, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
            )
            # Deskew
            coords = np.column_stack(np.where(thr == 0))
            if coords.size:
                rect = cv2.minAreaRect(coords)
                angle = rect[-1]
                angle = -(90 + angle) if angle < -45 else -angle
                (h, w) = thr.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                thr = cv2.warpAffine(
                    thr, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            return Image.fromarray(thr)
        except Exception as exc:
            log.debug("OpenCV preprocessing failed (%s); using Pillow fallback.", exc)

    # ---- Pillow fallback ----
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g, cutoff=1)
    try:
        g = g.filter(ImageFilter.UnsharpMask(radius=1.4, percent=140, threshold=6))
    except Exception:
        pass
    # Upscale helps Tesseract with small text
    g = g.resize((int(g.width * 1.5), int(g.height * 1.5)), Image.LANCZOS)
    g = g.point(lambda x: 255 if x > 180 else 0)
    return g


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def ocr_image(path: Path, lang: str = "eng") -> str:
    img = load_image(path)
    try:
        img = preprocess_for_ocr(img)
        text = pytesseract.image_to_string(img, lang=lang, config=TESS_CONFIG)
        return text
    finally:
        img.close()


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def extract_folder_name(text: str, patterns: list[str]) -> str | None:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def process_image(
    src: Path,
    output_dir: Path,
    patterns: list[str],
    lang: str,
    move: bool,
    save_ocr: bool,
) -> str:
    log.info("OCR: %s", src.name)
    try:
        text = ocr_image(src, lang=lang)
    except Exception as exc:
        log.error("Failed %s: %s", src.name, exc)
        return UNSORTED_DIR

    folder_name = extract_folder_name(text, patterns) or UNSORTED_DIR
    dest_dir = output_dir / folder_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / src.name
    # Avoid collision
    counter = 1
    while dest.exists():
        dest = dest_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1

    if move:
        shutil.move(str(src), dest)
        log.info("  Moved  → %s/", folder_name)
    else:
        shutil.copy2(str(src), dest)
        log.info("  Copied → %s/", folder_name)

    if save_ocr:
        base = dest.with_suffix("")
        base.with_suffix(".ocr.txt").write_text(text, encoding="utf-8")
        payload = {"file": src.name, "folder": folder_name, "ocr_text": text}
        base.with_suffix(".ocr.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return folder_name


def process_existing(
    input_dir: Path,
    output_dir: Path,
    patterns: list[str],
    lang: str,
    move: bool,
    save_ocr: bool,
) -> int:
    images = [p for p in input_dir.iterdir() if p.is_file() and is_image(p)]
    for img in images:
        process_image(img, output_dir, patterns, lang, move, save_ocr)
    log.info("Processed %d existing image(s).", len(images))
    return len(images)


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------

class _ImageHandler(FileSystemEventHandler):
    def __init__(self, output_dir, patterns, lang, move, save_ocr):
        self.output_dir = output_dir
        self.patterns = patterns
        self.lang = lang
        self.move = move
        self.save_ocr = save_ocr

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if not is_image(path):
            return
        # Small delay so the file is fully written before we open it
        time.sleep(0.5)
        process_image(
            path, self.output_dir,
            self.patterns, self.lang,
            self.move, self.save_ocr,
        )


def watch(
    input_dir: Path,
    output_dir: Path,
    patterns: list[str],
    lang: str,
    move: bool,
    save_ocr: bool,
) -> None:
    if not WATCHDOG_AVAILABLE:
        sys.exit(
            "Error: watchdog is required for --watch mode.\n"
            "Install: pip install watchdog"
        )
    handler = _ImageHandler(output_dir, patterns, lang, move, save_ocr)
    observer = Observer()
    observer.schedule(handler, str(input_dir), recursive=False)
    observer.start()
    log.info("Watching '%s' for new images … (Ctrl+C to stop)", input_dir)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Stopping watcher.")
        observer.stop()
    observer.join()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Read label text from images (incl. RAW via rawpy) "
            "and auto-sort them into folders."
        )
    )
    p.add_argument("input",  type=Path, help="Input folder with images")
    p.add_argument("output", type=Path, help="Output root folder")
    p.add_argument("--watch",    action="store_true", help="Watch input for new files")
    p.add_argument("--move",     action="store_true", help="Move files (default: copy)")
    p.add_argument("--save-ocr", action="store_true", help="Write .ocr.txt/.ocr.json sidecars")
    p.add_argument(
        "--pattern", dest="patterns", metavar="REGEX", action="append",
        help="Regex with one capture group = folder name. Repeat for multiple patterns. "
             "First match wins. Overrides built-in defaults.",
    )
    p.add_argument("--lang", default="eng", help="Tesseract language code (default: eng)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()

    if not input_dir.is_dir():
        sys.exit(f"Error: input folder not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns = args.patterns if args.patterns else DEFAULT_PATTERNS

    log.info("Input : %s", input_dir)
    log.info("Output: %s", output_dir)
    log.info("Mode  : %s | Lang: %s | OpenCV: %s | RAW: %s",
             "move" if args.move else "copy",
             args.lang,
             "yes" if CV2_AVAILABLE else "no (pip install opencv-python)",
             "yes" if RAWPY_AVAILABLE else "no (pip install rawpy)")

    # Always process files already in the folder
    process_existing(input_dir, output_dir, patterns, args.lang, args.move, args.save_ocr)

    if args.watch:
        watch(input_dir, output_dir, patterns, args.lang, args.move, args.save_ocr)


if __name__ == "__main__":
    main()