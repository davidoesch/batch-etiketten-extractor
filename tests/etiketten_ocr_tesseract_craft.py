#!/usr/bin/env python3
"""
Etiketten-OCR für bereits extrahierte Etiketten
Input: Nur das Etikett-Bild (bereits rotiert)
Output: 6 Felder mit CRAFT + Tesseract
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import pytesseract
from craft_text_detector import Craft


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EtikettenOCR:
    """OCR für bereits extrahierte Etiketten-Bilder."""

    def __init__(self, confidence_threshold: float = 0.5, debug_mode: bool = False):
        self.confidence_threshold = confidence_threshold
        self.debug_mode = debug_mode

        logger.info("Initialisiere CRAFT...")
        self.craft = Craft(
            output_dir=None,
            crop_type="poly",
            cuda=False,
            long_size=1280
        )
        logger.info("CRAFT bereit")

        self.tesseract_config = r'--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ0123456789/-. '
        logger.info("Tesseract konfiguriert für deutsche Großbuchstaben")

    def detect_text_lines(
        self, label: np.ndarray, debug_path: Path = None
    ) -> List[int]:
        """
        Detektiert Textzeilen und gibt Y-Positionen zurück.

        Args:
            label: Etikett-Bild
            debug_path: Debug-Verzeichnis

        Returns:
            Liste von Y-Positionen der Zeilentrennungen [y1, y2, ...]
        """
        logger.info("Detektiere Textzeilen mit CRAFT...")

        # Vorverarbeitung
        if len(label.shape) == 3:
            gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        else:
            gray = label

        # Leichte Vergrößerung für bessere Detektion
        scale = 1.5
        enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # RGB für CRAFT
        rgb = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2RGB)

        try:
            # CRAFT Textdetektion
            result = self.craft.detect_text(rgb)
            regions = result["boxes"]

            logger.info(f"CRAFT fand {len(regions)} Textregionen")

            if not regions:
                logger.warning("Keine Textregionen gefunden, verwende fixe 3 Zeilen")
                h = label.shape[0]
                return [h//3, 2*h//3]

            # Extrahiere Y-Positionen (Zentren)
            y_positions = []
            for region in regions:
                y_coords = [p[1] / scale for p in region]  # Zurück zu Original-Skala
                y_center = sum(y_coords) / len(y_coords)
                y_positions.append(y_center)

            y_positions.sort()

            logger.debug(f"Y-Positionen: {[f'{y:.1f}' for y in y_positions]}")

            # Gruppiere Y-Positionen in Zeilen
            # Texte in gleicher Zeile haben ähnliche Y-Werte
            rows = []
            current_row = [y_positions[0]]
            tolerance = label.shape[0] * 0.05  # 5% der Höhe als Toleranz

            for y in y_positions[1:]:
                if abs(y - current_row[-1]) <= tolerance:
                    # Gehört zur gleichen Zeile
                    current_row.append(y)
                else:
                    # Neue Zeile
                    rows.append(current_row)
                    current_row = [y]

            if current_row:
                rows.append(current_row)

            logger.info(f"Gefundene Textzeilen: {len(rows)}")

            # Berechne durchschnittliche Y-Position pro Zeile
            row_centers = [sum(row) / len(row) for row in rows]

            logger.debug(f"Zeilen-Zentren: {[f'{y:.1f}' for y in row_centers]}")

            # Berechne Trennlinien zwischen Zeilen
            separators = []
            for i in range(len(row_centers) - 1):
                separator = (row_centers[i] + row_centers[i+1]) / 2
                separators.append(int(separator))

            logger.info(f"Trennlinien bei Y: {separators}")

            # Debug-Visualisierung
            if debug_path:
                debug_img = label.copy()

                # Zeichne CRAFT-Regionen
                for region in regions:
                    pts = np.array([[p[0]/scale, p[1]/scale] for p in region], dtype=np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)

                # Zeichne Zeilen-Zentren
                for i, y_center in enumerate(row_centers):
                    cv2.line(debug_img, (0, int(y_center)), (debug_img.shape[1], int(y_center)),
                            (255, 0, 0), 2)
                    cv2.putText(debug_img, f"Zeile {i+1}", (10, int(y_center) - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Zeichne Trennlinien
                for sep in separators:
                    cv2.line(debug_img, (0, sep), (debug_img.shape[1], sep),
                            (0, 0, 255), 3)

                cv2.imwrite(str(debug_path / "01_detected_rows.jpg"), debug_img)

            return separators

        except Exception as e:
            logger.error(f"Fehler bei Zeilen-Detektion: {e}")
            # Fallback: fixe 3 Zeilen
            h = label.shape[0]
            return [h//3, 2*h//3]

    def split_into_cells(
        self, label: np.ndarray, debug_path: Path = None
    ) -> List[np.ndarray]:
        """
        Teilt Etikett in Zeilen (dynamisch) × 2 Spalten auf.

        Args:
            label: Etikett-Bild
            debug_path: Debug-Verzeichnis

        Returns:
            Liste von Zellen [row1_col1, row1_col2, row2_col1, ...]
        """
        h, w = label.shape[:2]

        logger.info(f"Teile Etikett {w}×{h} auf")

        # Detektiere Zeilen dynamisch
        row_separators = self.detect_text_lines(label, debug_path)

        # Erstelle Zeilen-Grenzen
        row_boundaries = [0] + row_separators + [h]
        num_rows = len(row_boundaries) - 1

        logger.info(f"Zeilen: {num_rows}, Spalten: 2")

        if debug_path:
            # Zeichne Grid
            debug_img = label.copy()

            # Horizontale Linien (Zeilen-Trennungen)
            for sep in row_separators:
                cv2.line(debug_img, (0, sep), (w, sep), (0, 255, 0), 3)

            # Vertikale Linie (Spalten-Trennung)
            cell_w = w // 2
            cv2.line(debug_img, (cell_w, 0), (cell_w, h), (0, 255, 0), 3)

            # Beschrifte Zellen
            for row in range(num_rows):
                for col in range(2):
                    y = (row_boundaries[row] + row_boundaries[row+1]) // 2
                    x = col * cell_w + cell_w // 2
                    cv2.putText(debug_img, f"R{row+1}C{col+1}", (x-40, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imwrite(str(debug_path / "02_grid.jpg"), debug_img)

        # Extrahiere Zellen
        cells = []
        cell_width = w // 2

        for row_idx in range(num_rows):
            y_start = row_boundaries[row_idx]
            y_end = row_boundaries[row_idx + 1]

            for col in range(2):
                x_start = col * cell_width
                x_end = (col + 1) * cell_width if col < 1 else w

                cell = label[y_start:y_end, x_start:x_end]
                cells.append(cell)

                if debug_path:
                    cv2.imwrite(
                        str(debug_path / f"03_cell_r{row_idx+1}_c{col+1}.jpg"),
                        cell
                    )

        logger.info(f"Extrahiert: {len(cells)} Zellen")
        return cells

    def preprocess_cell(self, cell: np.ndarray) -> np.ndarray:
        """Bereitet Zelle für CRAFT vor."""
        if len(cell.shape) == 3:
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell

        # Vergrößern (3x)
        scale = 3
        enlarged = cv2.resize(gray, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)

        # Entrauschen
        denoised = cv2.fastNlMeansDenoising(enlarged, h=10)

        # Schärfen
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # Kontrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(sharpened)

        return enhanced

    def detect_text_craft(
        self, cell: np.ndarray, cell_idx: int, debug_path: Path = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Detektiert Textregionen mit CRAFT."""
        preprocessed = self.preprocess_cell(cell)

        # RGB für CRAFT
        if len(preprocessed.shape) == 2:
            rgb = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
        else:
            rgb = preprocessed

        try:
            result = self.craft.detect_text(rgb)
            regions = result["boxes"]

            logger.info(f"  Zelle {cell_idx+1}: {len(regions)} Textregionen")

            if debug_path and regions:
                debug_img = rgb.copy()
                for i, region in enumerate(regions):
                    pts = np.array(region, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(debug_img, [pts], True, (0, 255, 0), 3)
                    cv2.putText(debug_img, str(i), tuple(region[0]),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imwrite(str(debug_path / f"04_craft_cell{cell_idx+1}.jpg"), debug_img)

            # Extrahiere Regionen
            text_regions = []
            for region in regions:
                x_coords = [p[0] for p in region]
                y_coords = [p[1] for p in region]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))

                padding = 5
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(rgb.shape[1], x_max + padding)
                y_max = min(rgb.shape[0], y_max + padding)

                cropped = preprocessed[y_min:y_max, x_min:x_max]

                if cropped.size > 0:
                    text_regions.append((region, cropped))

            return text_regions

        except Exception as e:
            logger.error(f"  CRAFT-Fehler Zelle {cell_idx+1}: {e}")
            return []

    def ocr_tesseract(self, region: np.ndarray) -> Tuple[str, float]:
        """OCR mit Tesseract."""
        # Binarisierung
        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            data = pytesseract.image_to_data(
                binary, config=self.tesseract_config,
                output_type=pytesseract.Output.DICT, lang='deu'
            )

            texts = []
            confs = []

            for i, text in enumerate(data['text']):
                text = text.strip()
                conf = float(data['conf'][i])

                if text and conf > 0:
                    texts.append(text.upper())
                    confs.append(conf)

            if texts:
                combined = ' '.join(texts)
                avg_conf = sum(confs) / len(confs) / 100.0
                return combined, avg_conf
            else:
                return '', 0.0

        except Exception as e:
            logger.error(f"    Tesseract-Fehler: {e}")
            return '', 0.0

    def process_cell(
        self, cell: np.ndarray, cell_idx: int, debug_path: Path = None
    ) -> Dict[str, any]:
        """Verarbeitet eine Zelle."""
        logger.info(f"Verarbeite Zelle {cell_idx+1}...")

        # CRAFT Detektion
        regions = self.detect_text_craft(cell, cell_idx, debug_path)

        if not regions:
            logger.warning(f"  Zelle {cell_idx+1}: Keine Textregionen")
            return {'text': '', 'confidence': 0.0}

        # Tesseract OCR
        all_texts = []
        all_confs = []

        for region_idx, (bbox, region_img) in enumerate(regions):
            text, conf = self.ocr_tesseract(region_img)
            if text:
                all_texts.append(text)
                all_confs.append(conf)
                logger.debug(f"    Region {region_idx}: '{text}' ({conf:.2f})")

        if all_texts:
            combined = ' '.join(all_texts)
            avg_conf = sum(all_confs) / len(all_confs)
            logger.info(f"  ✓ Zelle {cell_idx+1}: '{combined}' ({avg_conf:.2f})")
            return {'text': combined, 'confidence': avg_conf}
        else:
            logger.warning(f"  ✗ Zelle {cell_idx+1}: Kein Text erkannt")
            return {'text': '', 'confidence': 0.0}

    def process_label(self, label_path: Path) -> Dict[str, any]:
        """Verarbeitet ein Etikett-Bild."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Verarbeite: {label_path.name}")
        logger.info(f"{'='*60}")

        try:
            label = cv2.imread(str(label_path))
            if label is None:
                logger.error(f"Konnte nicht laden: {label_path}")
                return None

            logger.info(f"Etikett geladen: {label.shape[1]}×{label.shape[0]}")

            # Debug
            debug_path = None
            if self.debug_mode:
                debug_path = label_path.parent / f"debug_{label_path.stem}"
                debug_path.mkdir(exist_ok=True)
                cv2.imwrite(str(debug_path / "00_input.jpg"), label)

            # Split in Zellen
            cells = self.split_into_cells(label, debug_path)

            # Verarbeite Zellen
            fields = []
            for i, cell in enumerate(cells):
                field = self.process_cell(cell, i, debug_path)
                fields.append(field)

            # Konfidenz
            confidences = [f['confidence'] for f in fields if f['text']]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            result = {
                'filename': label_path.name,
                'fields': fields,
                'num_rows': len(fields) // 2,  # Anzahl Zeilen
                'avg_confidence': avg_conf,
                'manual_check': avg_conf < self.confidence_threshold
            }

            logger.info(f"\n{'='*60}")
            logger.info(f"ERGEBNIS: {label_path.name}")
            logger.info(f"{'='*60}")
            for i, field in enumerate(fields, 1):
                status = "✓" if field['confidence'] >= self.confidence_threshold else "✗"
                logger.info(f"{status} Feld {i}: '{field['text']}' ({field['confidence']:.2f})")
            logger.info(f"Durchschnitt: {avg_conf:.2f}")
            logger.info(f"Manuelle Prüfung: {'JA' if result['manual_check'] else 'NEIN'}")
            logger.info(f"{'='*60}\n")

            return result

        except Exception as e:
            logger.error(f"Fehler: {e}", exc_info=True)
            return None

    def process_directory(self, input_dir: Path, output_csv: Path):
        """Verarbeitet alle Etikett-Bilder."""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        images = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

        if not images:
            logger.error(f"Keine Bilder in {input_dir}")
            return

        logger.info(f"Gefunden: {len(images)} Etiketten-Bilder")

        results = []
        for img_path in images:
            result = self.process_label(img_path)
            if result:
                results.append(result)

        # CSV
        self._write_csv(results, output_csv)

        # Zusammenfassung
        total = len(results)
        manual = sum(1 for r in results if r['manual_check'])
        logger.info(f"\n{'='*60}")
        logger.info(f"ZUSAMMENFASSUNG")
        logger.info(f"{'='*60}")
        logger.info(f"Verarbeitet: {total}")
        logger.info(f"Erfolgreich: {total - manual}")
        logger.info(f"Manuelle Prüfung: {manual}")
        logger.info(f"CSV: {output_csv}")
        logger.info(f"{'='*60}")

    def _write_csv(self, results: List[Dict], output_csv: Path):
        """Schreibt CSV mit variabler Anzahl von Feldern."""
        # Finde maximale Anzahl von Feldern
        max_fields = max(len(r['fields']) for r in results) if results else 6

        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['filename', 'num_rows']
            for i in range(1, max_fields + 1):
                fieldnames.extend([f'field{i}', f'field{i}_score'])
            fieldnames.append('manual_check')

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    'filename': result['filename'],
                    'num_rows': result.get('num_rows', len(result['fields']) // 2)
                }

                for i, field in enumerate(result['fields'], 1):
                    row[f'field{i}'] = field['text']
                    row[f'field{i}_score'] = f"{field['confidence']:.4f}"

                # Fülle fehlende Felder auf
                for i in range(len(result['fields']) + 1, max_fields + 1):
                    row[f'field{i}'] = ''
                    row[f'field{i}_score'] = '0.0000'

                row['manual_check'] = 'JA' if result['manual_check'] else 'NEIN'
                writer.writerow(row)

        logger.info(f"CSV geschrieben: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='Etiketten-OCR für bereits extrahierte Etiketten'
    )
    parser.add_argument('input_dir', type=str, help='Verzeichnis mit Etikett-Bildern')
    parser.add_argument('--output-dir', type=str, default='.', help='Ausgabeverzeichnis')
    parser.add_argument('--output-csv', type=str, default='etiketten_text.csv')
    parser.add_argument('--confidence', type=float, default=0.5, help='Konfidenz-Schwelle')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    parser.add_argument('--debug', action='store_true', help='Debug-Modus')

    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Ungültiges Verzeichnis: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Etiketten-OCR (CRAFT + Tesseract)")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir / args.output_csv}")

    ocr = EtikettenOCR(confidence_threshold=args.confidence, debug_mode=args.debug)
    ocr.process_directory(input_dir, output_dir / args.output_csv)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv = [
            'etiketten_ocr.py',
            '/home/menas/Downloads/nade/best/',
            '--output-dir', '/home/menas/Downloads/nade/results',
            '--debug',
            '--log-level', 'INFO'
        ]

    main()