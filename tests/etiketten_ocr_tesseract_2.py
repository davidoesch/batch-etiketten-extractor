#!/usr/bin/env python3
"""
Etiketten-OCR-Extraktor (Optimiert für Nadeldrucker)
Extrahiert Text von Nadeldrucker-Etiketten mittels geometrischem Crop und Tesseract.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Logging-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Tesseract Pfad (Falls Windows genutzt wird, hier entkommentieren und anpassen)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class EtikettenOCR:
    """Klasse zur Verarbeitung von Etiketten-Photos mit OCR."""

    def __init__(self, confidence_threshold: float = 40.0, debug_mode: bool = False):
        """
        Args:
            confidence_threshold: Tesseract Score (0-100). Alles unter 40 ist meist Rauschen.
            debug_mode: Aktiviert das Speichern von Zwischenbildern.
        """
        self.confidence_threshold = confidence_threshold
        self.debug_mode = debug_mode
        logger.info("Initialisiere Tesseract Konfiguration...")
        # PSM 6 = Assume a single uniform block of text (Wichtig für Spalten!)
        self.tess_config = r'--oem 3 --psm 6'

    def extract_roi_geometrically(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Schneidet den rechten Bereich (ca. 18-20%) des Bildes aus und rotiert ihn.
        Verlässt sich auf die physische Konsistenz der Dias.
        """
        h, w = image.shape[:2]

        # Wir nehmen an, das Etikett ist in den rechten 18% des Bildes
        # Falls Text abgeschnitten wird, ändere 0.82 auf 0.78
        crop_start_x = int(w * 0.82)

        roi = image[0:h, crop_start_x:w]

        # Rotation 90 Grad gegen Uhrzeigersinn
        roi_rotated = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return roi_rotated, True

    def preprocess_roi(self, roi: np.ndarray, debug_path: Optional[Path] = None) -> np.ndarray:
        """
        THE SECRET SAUCE: Optimiert Nadeldrucker-Punkte zu soliden Linien.
        """
        # 1. Graustufen
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 2. Upscaling (Tesseract mag große Schrift)
        scale = 3
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # 3. Binarisierung (Otsu) -> Text wird Weiß, Hintergrund Schwarz
        # Wir nutzen Threshold Binary Inverse, damit Text SCHWARZ und Hintergrund WEISS wird (Besser für OCR)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Falls das Bild "Invertiert" ist (Weißer Text auf Schwarz), müssen wir es drehen.
        # Tesseract braucht Schwarz auf Weiß.
        # Wir zählen Pixel: Wenn mehr als 50% schwarz sind, ist es vermutlich dunkler Hintergrund.
        white_pixels = cv2.countNonZero(binary)
        total_pixels = binary.size
        if white_pixels < (total_pixels * 0.5):
            # Bild ist größtenteils schwarz -> Invertieren, damit Hintergrund weiß wird
            binary = cv2.bitwise_not(binary)

        # --- THE SECRET SAUCE: MORPHOLOGICAL OPERATIONS ---
        # Ziel: Die schwarzen Punkte (Text) sollen "ausbluten" und sich verbinden.
        # Da der Text jetzt SCHWARZ ist (Wert 0) und Hintergrund WEISS (255):
        # Erode auf den Hintergrund = Dilate auf den Text.

        # Kernel Größe: 4x4 ist aggressiv, verbindet aber gut.
        # Bei zu viel "Matsch" auf (3,3) reduzieren.
        kernel = np.ones((4,4), np.uint8)
        processed = cv2.erode(binary, kernel, iterations=1)

        if debug_path:
            cv2.imwrite(str(debug_path / "03_processed_sauce.jpg"), processed)

        return processed

    def extract_fields(self, processed_img: np.ndarray) -> List[Dict[str, any]]:
        """
        Führt Tesseract OCR aus und gruppiert die Ergebnisse in Zeilen.
        """
        # OCR Ausführen
        d = pytesseract.image_to_data(processed_img, lang='deu', config=self.tess_config, output_type=Output.DICT)

        lines_map = {}
        n_boxes = len(d['text'])

        for i in range(n_boxes):
            text = d['text'][i].strip()
            conf = int(d['conf'][i])

            # Filter: Leere Texte oder sehr niedrige Konfidenz
            if conf > 30 and len(text) > 0:
                # Gruppierung nach Y-Position (top)
                y = d['top'][i]

                # Suche, ob eine Zeile in der Nähe (±20px) existiert
                found_line = False
                for line_y in lines_map.keys():
                    if abs(line_y - y) < 20:
                        lines_map[line_y]['text'].append(text)
                        lines_map[line_y]['confs'].append(conf)
                        found_line = True
                        break

                if not found_line:
                    lines_map[y] = {'text': [text], 'confs': [conf]}

        # Sortiere Zeilen von Oben nach Unten
        sorted_ys = sorted(lines_map.keys())

        final_fields = []
        for y in sorted_ys:
            full_text = " ".join(lines_map[y]['text'])
            avg_conf = int(np.mean(lines_map[y]['confs']))

            # Heuristik: Ignoriere extrem kurze Fragmente (Rauschen am Rand)
            # Ausnahme: "D/88" ist kurz, aber wichtig.
            if len(full_text) > 2:
                final_fields.append({'text': full_text, 'confidence': avg_conf})

        # Padding: Auffüllen auf 6 Felder
        while len(final_fields) < 6:
            final_fields.append({'text': '', 'confidence': 0.0})

        return final_fields[:6]

    def process_image(self, image_path: Path) -> Optional[Dict[str, any]]:
        try:
            # Debug Pfad
            debug_path = None
            if self.debug_mode:
                debug_path = image_path.parent / f"debug_{image_path.stem}"
                debug_path.mkdir(exist_ok=True)

            image = cv2.imread(str(image_path))
            if image is None:
                return None

            # 1. Geometrischer Crop & Rotation
            roi, _ = self.extract_roi_geometrically(image)
            if debug_path:
                cv2.imwrite(str(debug_path / "01_roi_raw.jpg"), roi)

            # 2. Preprocessing (Secret Sauce)
            processed = self.preprocess_roi(roi, debug_path)

            # 3. Extraktion
            fields = self.extract_fields(processed)

            # Berechne Durchschnittskonfidenz der gefundenen Felder
            valid_confs = [f['confidence'] for f in fields if f['text']]
            avg_conf = sum(valid_confs) / len(valid_confs) if valid_confs else 0

            return {
                'filename': image_path.name,
                'fields': fields,
                'avg_confidence': avg_conf,
                'manual_check': avg_conf < self.confidence_threshold
            }

        except Exception as e:
            logger.error(f"Fehler bei {image_path.name}: {e}")
            return None

    def process_directory(self, input_dir: Path, output_csv: Path):
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG'}
        image_files = [f for f in input_dir.iterdir() if f.suffix in image_extensions]

        logger.info(f"Gefundene Bilder: {len(image_files)}")

        results = []
        for img_path in image_files:
            logger.info(f"Verarbeite {img_path.name}...")
            res = self.process_image(img_path)
            if res:
                results.append(res)

        self._write_csv(results, output_csv)

    def _write_csv(self, results: List[Dict], output_csv: Path):
        with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
            # Dynamische Header-Generierung basierend auf max gefundenen Feldern (immer 6)
            headers = ['filename']
            for i in range(1, 7):
                headers.extend([f'field{i}', f'field{i}_score'])
            headers.append('manual_check')

            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for res in results:
                row = {'filename': res['filename']}
                for i, field in enumerate(res['fields']):
                    row[f'field{i+1}'] = field['text']
                    row[f'field{i+1}_score'] = field['confidence']

                row['manual_check'] = "JA" if res['manual_check'] else ""
                writer.writerow(row)

        logger.info(f"Fertig! Datei gespeichert unter: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='OCR für Nadeldrucker-Etiketten')
    parser.add_argument('input_dir', type=str, help='Ordner mit Bildern')
    parser.add_argument('--confidence', type=float, default=40.0, help='Konfidenz-Schwelle')
    parser.add_argument('--debug', action='store_true', help='Speichert Debug-Bilder')

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_csv = input_path / 'etiketten_output.csv'

    ocr = EtikettenOCR(confidence_threshold=args.confidence, debug_mode=args.debug)
    ocr.process_directory(input_path, output_csv)

if __name__ == '__main__':
    # Standardwerte für schnellen Start ohne Argumente (optional)
    if len(sys.argv) == 1:
        # Beispielpfad - bitte anpassen
        sys.argv.append('/home/menas/Downloads/nade/best')
        sys.argv.append('--debug')

    main()