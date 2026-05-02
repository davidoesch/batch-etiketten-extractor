#!/usr/bin/env python3
"""
Batch-Etiketten-Extractor
=========================

Extrahiert mit Nadeldrucker erzeugte Etiketten von Dia-Rahmen-Fotos.

Jedes Etikett enthaelt sechs Felder, angeordnet in 3 Zeilen x 2 Spalten.
Das Etikett ist auf dem Rahmen um 90 Grad im Uhrzeigersinn gedreht und
wird daher vor der OCR um 90 Grad gegen den Uhrzeigersinn zurueckgedreht.

Aufruf (Windows DOS / bash):
    python extract_labels.py <input_dir> -o labels.csv
    python extract_labels.py <input_dir> -o labels.csv --debug-dir debug

Abhaengigkeiten:
    pip install opencv-python numpy pytesseract
    Zusaetzlich: Tesseract-Binary installieren und Sprachpaket 'deu'
      Windows: https://github.com/UB-Mannheim/tesseract/wiki
      Linux:   apt-get install tesseract-ocr tesseract-ocr-deu
"""

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract

# Auf Windows ggf. expliziten Pfad setzen, falls Tesseract nicht im PATH:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("extract_labels")


# --------------------------------------------------------------------- #
# 1. Etikett lokalisieren
# --------------------------------------------------------------------- #

def find_label_bbox(image_bgr):
    """
    Sucht das helle, laengliche Rechteck (Papieretikett) im Bild.
    Rueckgabe: (x, y, w, h) oder None, falls kein Kandidat gefunden wird.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Helle Bereiche extrahieren (Papier)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Luecken schliessen, damit das Etikett als eine Komponente erscheint
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    img_h, img_w = gray.shape[:2]
    img_area = img_h * img_w

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        longer = max(w, h)
        shorter = max(1, min(w, h))
        aspect = longer / shorter

        # Etikett ist deutlich laenglich und von mittlerer Groesse
        if aspect < 3.0:
            continue
        if area < img_area * 0.003:
            continue
        if area > img_area * 0.25:
            continue

        candidates.append((x, y, w, h, aspect, area))

    if not candidates:
        return None

    # Bevorzugt das laengste, danach das groesste Rechteck
    candidates.sort(key=lambda c: (c[4], c[5]), reverse=True)
    x, y, w, h, _, _ = candidates[0]
    return x, y, w, h


# --------------------------------------------------------------------- #
# 2. Ausrichten und Vorverarbeiten
# --------------------------------------------------------------------- #

def ensure_landscape(label_bgr):
    """Hochformat-Etiketten um 90 Grad gegen den Uhrzeigersinn drehen."""
    h, w = label_bgr.shape[:2]
    if h > w:
        return cv2.rotate(label_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return label_bgr


def preprocess_for_ocr(label_bgr, upscale=4):
    """
    Binaerbild vorbereiten, das fuer Nadeldruck-OCR geeignet ist.
    Hochskalieren, weichzeichnen, Otsu, morphologische Schliessung.
    """
    gray = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2GRAY)

    # Hochskalieren, damit einzelne Druckpunkte zu Zeichenflaechen verschmelzen
    gray = cv2.resize(
        gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
    )

    # Leichte Weichzeichnung vereinigt benachbarte Punkte eines Zeichens
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu-Schwellwert kompensiert Helligkeitsunterschiede der Bildserie
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Luecken innerhalb der Zeichen schliessen
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


# --------------------------------------------------------------------- #
# 3. OCR und Feldzerlegung
# --------------------------------------------------------------------- #

TESS_CONFIG = "--oem 3 --psm 6 -l deu"


def ocr_label(binary_img):
    """Rohen OCR-Text des Etiketts zurueckgeben."""
    return pytesseract.image_to_string(binary_img, config=TESS_CONFIG)


def split_into_six(raw_text):
    """
    OCR-Ausgabe in sechs Felder (3 Zeilen x 2 Spalten) zerlegen.
    Trenner innerhalb einer Zeile: zwei oder mehr Leerzeichen bzw. Tab.
    Fehlende Zellen werden mit Leerstrings aufgefuellt.
    """
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    fields = []
    for line in lines[:3]:
        parts = re.split(r"\s{2,}|\t+", line, maxsplit=1)
        if len(parts) == 2:
            fields.extend([parts[0].strip(), parts[1].strip()])
        else:
            fields.extend([line, ""])
    while len(fields) < 6:
        fields.append("")
    return fields[:6]


# --------------------------------------------------------------------- #
# 4. Pipeline pro Bild
# --------------------------------------------------------------------- #

def process_image(image_path, debug_dir=None):
    """
    Vollstaendige Verarbeitung eines einzelnen Bildes.
    Rueckgabe: (Liste der sechs Felder, Rohtext aus OCR).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning("Bild nicht lesbar: %s", image_path.name)
        return ["", "", "", "", "", ""], ""

    bbox = find_label_bbox(img)
    if bbox is None:
        logger.warning("Kein Etikett erkannt in: %s", image_path.name)
        return ["", "", "", "", "", ""], ""

    x, y, w, h = bbox
    pad = 8
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img.shape[1], x + w + pad)
    y1 = min(img.shape[0], y + h + pad)
    crop = img[y0:y1, x0:x1]

    crop = ensure_landscape(crop)
    binary = preprocess_for_ocr(crop)

    raw = ocr_label(binary)
    fields = split_into_six(raw)

    if debug_dir is not None:
        stem = image_path.stem
        cv2.imwrite(str(debug_dir / f"{stem}_crop.png"), crop)
        cv2.imwrite(str(debug_dir / f"{stem}_binary.png"), binary)
        (debug_dir / f"{stem}.txt").write_text(raw, encoding="utf-8")

    return fields, raw


# --------------------------------------------------------------------- #
# 5. Einstiegspunkt
# --------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Extrahiert sechsfeldrige Nadeldruck-Etiketten in eine CSV."
    )
    parser.add_argument(
        "input_dir",
        help="Verzeichnis mit JPG/PNG-Aufnahmen der Dia-Rahmen",
    )
    parser.add_argument(
        "-o", "--output",
        default="labels.csv",
        help="Ausgabe-CSV (Standard: labels.csv)",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Optionales Verzeichnis fuer Zuschnitte, Binaerbilder und Rohtext",
    )
    parser.add_argument(
        "--delimiter",
        default=";",
        help="CSV-Trenner (Standard: ';')",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        logger.error("Eingabeverzeichnis nicht gefunden: %s", input_dir)
        sys.exit(1)

    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    )
    logger.info("Gefunden: %d Bilder in %s", len(image_files), input_dir)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=args.delimiter)
        writer.writerow([
            "filename",
            "field1", "field2",
            "field3", "field4",
            "field5", "field6",
        ])
        for image_path in image_files:
            logger.info("Verarbeite %s", image_path.name)
            fields, _raw = process_image(image_path, debug_dir=debug_dir)
            writer.writerow([image_path.name] + fields)

    logger.info("CSV geschrieben: %s", args.output)


if __name__ == "__main__":
    main()