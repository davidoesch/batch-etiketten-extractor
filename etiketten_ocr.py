#!/usr/bin/env python3
"""
Etiketten-OCR-Extraktor
Extrahiert Text von Nadeldrucker-Etiketten aus Photos mit automatischer Erkennung.
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import easyocr


# Logging-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EtikettenOCR:
    """Klasse zur Verarbeitung von Etiketten-Photos mit OCR."""

    def __init__(self, confidence_threshold: float = 0.5, debug_mode: bool = False):
        """
        Initialisiert den OCR-Reader.

        Args:
            confidence_threshold: Minimale Konfidenz für OCR-Ergebnisse (0.0-1.0)
            debug_mode: Aktiviert Debug-Ausgaben und speichert Zwischenbilder
        """
        self.confidence_threshold = confidence_threshold
        self.debug_mode = debug_mode
        self._cached_text_results = None
        self._cached_text_bbox = None
        logger.info("Initialisiere EasyOCR für Deutsch...")
        self.reader = easyocr.Reader(['de'], gpu=True)
        logger.info("EasyOCR erfolgreich initialisiert")

    def rotate_image(self, image: np.ndarray) -> np.ndarray:
        """
        Rotiert das Bild 90 Grad nach links (gegen den Uhrzeigersinn).

        Args:
            image: Eingabebild

        Returns:
            Rotiertes Bild
        """
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def find_label_contours(self, image: np.ndarray, debug_path: Optional[Path] = None) -> List[Tuple[int, int, int, int]]:
        """
        Findet Etiketten im gesamten Bild durch Textdetektion.

        Args:
            image: Eingabebild (RGB)
            debug_path: Optionaler Pfad zum Speichern von Debug-Bildern

        Returns:
            Liste von Bounding Boxes (x, y, w, h) für gefundene Etiketten
        """
        img_height, img_width = image.shape[:2]

        logger.debug(f"Bildgröße: {img_width}x{img_height}")
        logger.debug(f"Suche Text im gesamten Bild")

        # Verwende EasyOCR zur Textdetektion im GESAMTEN Bild
        logger.info("Führe Textdetektion im gesamten Bild durch...")
        text_results = self.reader.readtext(image, detail=1)

        logger.info(f"Gefundene Textblöcke: {len(text_results)}")

        if not text_results:
            logger.warning("Keine Textblöcke im Bild gefunden")
            return []

        # Debug: Visualisiere gefundene Textblöcke
        if debug_path:
            debug_text = image.copy()
            for i, (bbox, text, conf) in enumerate(text_results):
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(debug_text, [pts], True, (0, 255, 0), 2)
                cv2.putText(debug_text, f"{i}:{text[:15]}", tuple(pts[0]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.imwrite(str(debug_path / "03_detected_text.jpg"), debug_text)

        # Finde Bounding Box aller Textblöcke zusammen
        all_points = []
        for bbox, text, conf in text_results:
            for point in bbox:
                all_points.append(point)

        if not all_points:
            return []

        all_points = np.array(all_points, dtype=np.int32)

        # Berechne Bounding Box um alle Textblöcke
        x_min = int(np.min(all_points[:, 0]))
        y_min = int(np.min(all_points[:, 1]))
        x_max = int(np.max(all_points[:, 0]))
        y_max = int(np.max(all_points[:, 1]))

        # Füge Padding hinzu
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img_width, x_max + padding)
        y_max = min(img_height, y_max + padding)

        w = x_max - x_min
        h = y_max - y_min

        logger.info(f"Etiketten-Box gefunden: Position ({x_min},{y_min}), Größe {w}x{h}")

        # Debug: Zeige finale Box
        if debug_path:
            debug_final = image.copy()
            cv2.rectangle(debug_final, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)
            cv2.putText(debug_final, "Etikette gefunden", (x_min, y_min-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.imwrite(str(debug_path / "05_detected_labels.jpg"), debug_final)

        # Speichere die Textresultate für spätere Verwendung
        self._cached_text_results = text_results
        self._cached_text_bbox = (x_min, y_min, x_max, y_max)

        # Rückgabe als Liste mit einer Box (Koordinaten im Gesamtbild)
        return [(x_min, y_min, w, h)]

    def preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Bereitet ROI für OCR vor: Optimiert für Nadeldrucker-Text.

        Args:
            roi: Region of Interest (Etikett)

        Returns:
            Vorverarbeitetes Bild
        """
        # Graustufen
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi

        # Vergrößere Bild für bessere OCR (3x größer)
        scale_factor = 3
        height, width = gray.shape
        enlarged = cv2.resize(gray, (width * scale_factor, height * scale_factor),
                             interpolation=cv2.INTER_CUBIC)

        logger.debug(f"Bild vergrößert: {width}x{height} -> {enlarged.shape[1]}x{enlarged.shape[0]}")

        # Entrauschen
        denoised = cv2.fastNlMeansDenoising(enlarged, h=10)

        # Schärfen für klarere Kanten (wichtig für Nadeldrucker-Punkte)
        kernel_sharpen = np.array([[-1,-1,-1],
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)

        # Kontrast erhöhen mit CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(sharpened)

        # Adaptive Schwellenwertbildung - optimiert für Nadeldrucker
        # Größerer Block für dickere Nadeldrucker-Punkte
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5
        )

        # Leichte morphologische Operation um Nadeldruckerpunkte zu verbinden
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        logger.debug("ROI vorverarbeitet: vergrößert, geschärft, binarisiert")

        return binary

    def extract_fields_from_label(
        self, roi: np.ndarray, debug_path: Optional[Path] = None
    ) -> List[Dict[str, any]]:
        """
        Extrahiert die 6 Felder (3 Zeilen × 2 Spalten) aus einer Etikette.
        Verwendet die bereits gecachten Texterkennungsergebnisse.

        Args:
            roi: Vorverarbeitetes Etiketten-Bild (wird für Debug verwendet)
            debug_path: Optionaler Pfad für Debug-Ausgaben

        Returns:
            Liste von Dictionaries mit 'text' und 'confidence' für jedes Feld
        """
        logger.debug("Extrahiere Felder aus gecachten OCR-Ergebnissen...")

        # Verwende die bereits erkannten Texte aus find_label_contours
        if not self._cached_text_results:
            logger.warning("Keine gecachten OCR-Ergebnisse verfügbar")
            return [{'text': '', 'confidence': 0.0} for _ in range(6)]

        results = self._cached_text_results
        x_min, y_min, x_max, y_max = self._cached_text_bbox

        logger.debug(f"Verwende {len(results)} gecachte Textblöcke")

        # Filtere nur Texte, die innerhalb der Etiketten-Box liegen
        filtered_results = []
        for bbox, text, conf in results:
            # Berechne Zentrum des Textblocks
            center_x = sum([p[0] for p in bbox]) / 4
            center_y = sum([p[1] for p in bbox]) / 4

            # Prüfe ob Zentrum in der Etiketten-Box liegt
            if x_min <= center_x <= x_max and y_min <= center_y <= y_max:
                filtered_results.append((bbox, text, conf))

        logger.debug(f"{len(filtered_results)} Textblöcke liegen in der Etiketten-Box")

        # Bereinige Texte: Nur Großbuchstaben, entferne Artefakte
        cleaned_results = []
        for bbox, text, conf in filtered_results:
            # Konvertiere zu Großbuchstaben
            text_clean = text.upper()

            # Entferne häufige OCR-Fehler bei Nadeldrucker
            text_clean = text_clean.replace('|', 'I')
            text_clean = text_clean.replace('!', 'I')

            # Entferne Leerzeichen am Anfang/Ende
            text_clean = text_clean.strip()

            if text_clean:  # Nur nicht-leere Texte
                # Passe Bbox-Koordinaten relativ zur Etiketten-Box an
                bbox_relative = [(p[0] - x_min, p[1] - y_min) for p in bbox]
                cleaned_results.append((bbox_relative, text_clean, conf))
                logger.debug(f"  Text: '{text_clean}' (Original: '{text}', Konfidenz: {conf:.2f})")

        # Debug: Visualisiere OCR-Ergebnisse auf ROI
        if debug_path and cleaned_results:
            debug_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR) if len(roi.shape) == 2 else roi.copy()
            for i, (bbox, text, conf) in enumerate(cleaned_results):
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)
                cv2.putText(debug_img, f"{i}:{text[:15]}", tuple(pts[0]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.imwrite(str(debug_path / "08_ocr_results_on_roi.jpg"), debug_img)
            logger.debug(f"OCR-Visualisierung auf ROI gespeichert")

        if not cleaned_results:
            logger.warning("Keine verwertbaren Texte nach Bereinigung")
            return [{'text': '', 'confidence': 0.0} for _ in range(6)]

        # Sortiere Ergebnisse nach Y-Position (oben nach unten)
        results_sorted = sorted(cleaned_results, key=lambda x: x[0][0][1])

        # Gruppiere in Zeilen (ähnliche Y-Koordinaten)
        rows = self._group_into_rows(results_sorted, roi.shape[0])

        logger.debug(f"Text in {len(rows)} Zeilen gruppiert")

        # Extrahiere bis zu 6 Felder (3 Zeilen, 2 Spalten)
        fields = []
        for row_idx, row in enumerate(rows[:3]):  # Maximal 3 Zeilen
            # Sortiere Elemente der Zeile nach X-Position
            row_sorted = sorted(row, key=lambda x: x[0][0][0])

            logger.debug(f"Zeile {row_idx+1}: {len(row_sorted)} Elemente")

            # Extrahiere bis zu 2 Spalten pro Zeile
            for i in range(2):
                if i < len(row_sorted):
                    text = row_sorted[i][1]
                    confidence = row_sorted[i][2]
                    fields.append({'text': text, 'confidence': confidence})
                    logger.debug(f"  Spalte {i+1}: '{text}' (Konfidenz: {confidence:.2f})")
                else:
                    fields.append({'text': '', 'confidence': 0.0})
                    logger.debug(f"  Spalte {i+1}: leer")

        # Fülle auf 6 Felder auf, falls weniger gefunden wurden
        while len(fields) < 6:
            fields.append({'text': '', 'confidence': 0.0})

        return fields[:6]

    def _group_into_rows(
        self, results: List, img_height: int, tolerance: int = 30
    ) -> List[List]:
        """
        Gruppiert OCR-Ergebnisse in Zeilen basierend auf Y-Koordinaten.

        Args:
            results: OCR-Ergebnisse
            img_height: Bildhöhe für relative Toleranz
            tolerance: Pixel-Toleranz für Zeilenzugehörigkeit

        Returns:
            Liste von Zeilen, jede Zeile enthält OCR-Ergebnisse
        """
        if not results:
            return []

        rows = []
        current_row = [results[0]]
        current_y = results[0][0][0][1]

        for result in results[1:]:
            y = result[0][0][1]

            # Prüfe ob Element zur aktuellen Zeile gehört
            if abs(y - current_y) <= tolerance:
                current_row.append(result)
            else:
                rows.append(current_row)
                current_row = [result]
                current_y = y

        # Füge letzte Zeile hinzu
        if current_row:
            rows.append(current_row)

        return rows

    def check_confidence(
        self, fields: List[Dict[str, any]]
    ) -> Tuple[bool, float]:
        """
        Prüft ob die Konfidenz aller Felder über dem Schwellenwert liegt.

        Args:
            fields: Liste von Feld-Dictionaries

        Returns:
            Tuple (Erfolg, durchschnittliche Konfidenz)
        """
        confidences = [f['confidence'] for f in fields if f['text']]

        if not confidences:
            return False, 0.0

        avg_confidence = sum(confidences) / len(confidences)
        success = avg_confidence >= self.confidence_threshold

        return success, avg_confidence

    def process_image(
        self, image_path: Path
    ) -> Optional[Dict[str, any]]:
        """
        Verarbeitet ein einzelnes Bild.

        Args:
            image_path: Pfad zum Bild

        Returns:
            Dictionary mit Ergebnissen oder None bei Fehler
        """
        logger.info(f"Verarbeite: {image_path.name}")

        try:
            # Lade Bild
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Konnte Bild nicht laden: {image_path}")
                return None

            logger.debug(f"Bild geladen: {image.shape[1]}x{image.shape[0]} Pixel")

            # Debug-Verzeichnis erstellen falls nötig
            debug_path = None
            if self.debug_mode:
                debug_path = image_path.parent / f"debug_{image_path.stem}"
                debug_path.mkdir(exist_ok=True)
                logger.debug(f"Debug-Modus aktiv, Ausgabe: {debug_path}")

            # Rotiere 90° nach links
            rotated = self.rotate_image(image)
            logger.debug(f"Bild rotiert: {rotated.shape[1]}x{rotated.shape[0]} Pixel")

            if debug_path:
                cv2.imwrite(str(debug_path / "00_rotated.jpg"), rotated)

            # Finde Etiketten-Konturen (führt OCR durch und cached Ergebnisse)
            label_boxes = self.find_label_contours(rotated, debug_path)

            if not label_boxes:
                logger.warning(f"Keine Etikette gefunden in {image_path.name}")
                return {
                    'filename': image_path.name,
                    'fields': [{'text': '', 'confidence': 0.0} for _ in range(6)],
                    'avg_confidence': 0.0,
                    'manual_check': True
                }

            # Verwende die größte gefundene Etikette
            x, y, w, h = label_boxes[0]
            logger.debug(f"Verwende Etikette bei Position ({x},{y}) mit Größe {w}x{h}")

            # Erweitere ROI leicht um Rand zu erfassen
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(rotated.shape[1] - x, w + 2*padding)
            h = min(rotated.shape[0] - y, h + 2*padding)

            roi = rotated[y:y+h, x:x+w]
            logger.debug(f"ROI extrahiert: {roi.shape[1]}x{roi.shape[0]} Pixel")

            if debug_path:
                cv2.imwrite(str(debug_path / "06_roi_extracted.jpg"), roi)

            # Vorverarbeitung (nur für Debug/Visualisierung, OCR bereits durchgeführt)
            preprocessed = self.preprocess_roi(roi)
            logger.debug("ROI vorverarbeitet")

            if debug_path:
                cv2.imwrite(str(debug_path / "07_preprocessed.jpg"), preprocessed)

            # Extrahiere Felder (verwendet gecachte OCR-Ergebnisse)
            fields = self.extract_fields_from_label(roi, debug_path)

            # Prüfe Konfidenz
            success, avg_confidence = self.check_confidence(fields)

            result = {
                'filename': image_path.name,
                'fields': fields,
                'avg_confidence': avg_confidence,
                'manual_check': not success
            }

            if not success:
                logger.warning(
                    f"Niedrige Konfidenz ({avg_confidence:.2f}) für "
                    f"{image_path.name} - Manuelle Prüfung erforderlich"
                )
            else:
                logger.info(
                    f"Erfolgreich verarbeitet: {image_path.name} "
                    f"(Konfidenz: {avg_confidence:.2f})"
                )

            # Log extrahierte Felder
            logger.debug("Extrahierte Felder:")
            for i, field in enumerate(fields, 1):
                logger.debug(f"  Feld {i}: '{field['text']}' (Konfidenz: {field['confidence']:.2f})")

            return result

        except Exception as e:
            logger.error(f"Fehler bei Verarbeitung von {image_path.name}: {e}", exc_info=True)
            return None

    def process_directory(
        self, input_dir: Path, output_csv: Path
    ) -> None:
        """
        Verarbeitet alle Bilder in einem Verzeichnis.

        Args:
            input_dir: Eingabeverzeichnis mit Bildern
            output_csv: Pfad zur Ausgabe-CSV-Datei
        """
        # Unterstützte Bildformate
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        # Finde alle Bilddateien
        image_files = [
            f for f in input_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            logger.error(f"Keine Bilddateien in {input_dir} gefunden")
            return

        logger.info(f"Gefundene Bilder: {len(image_files)}")

        # Verarbeite alle Bilder
        results = []
        for image_path in image_files:
            result = self.process_image(image_path)
            if result:
                results.append(result)

        # Schreibe CSV
        self._write_csv(results, output_csv)

        # Zusammenfassung
        total = len(results)
        manual_checks = sum(1 for r in results if r['manual_check'])
        logger.info(f"\n{'='*60}")
        logger.info(f"Verarbeitung abgeschlossen:")
        logger.info(f"  Gesamt verarbeitet: {total}")
        logger.info(f"  Erfolgreich: {total - manual_checks}")
        logger.info(f"  Manuelle Prüfung nötig: {manual_checks}")
        logger.info(f"  CSV erstellt: {output_csv}")
        logger.info(f"{'='*60}")

    def _write_csv(
        self, results: List[Dict[str, any]], output_csv: Path
    ) -> None:
        """
        Schreibt Ergebnisse in CSV-Datei.

        Args:
            results: Liste von Verarbeitungsergebnissen
            output_csv: Pfad zur Ausgabedatei
        """
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['filename']
            for i in range(1, 7):
                fieldnames.extend([f'field{i}', f'field{i}_score'])
            fieldnames.append('manual_check')

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {'filename': result['filename']}
                for i, field in enumerate(result['fields'], 1):
                    row[f'field{i}'] = field['text']
                    row[f'field{i}_score'] = f"{field['confidence']:.4f}"
                row['manual_check'] = 'JA' if result['manual_check'] else 'NEIN'

                writer.writerow(row)

        logger.info(f"CSV erfolgreich geschrieben: {output_csv}")


def main():
    """Hauptfunktion für Kommandozeilenaufruf."""
    parser = argparse.ArgumentParser(
        description='Extrahiert Text von Nadeldrucker-Etiketten aus Photos'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Verzeichnis mit Eingabe-Bildern'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Ausgabeverzeichnis für CSV (Standard: aktuelles Verzeichnis)'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        default='etiketten_text.csv',
        help='Name der Ausgabe-CSV-Datei (Standard: etiketten_text.csv)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Minimale Konfidenz-Schwelle (0.0-1.0, Standard: 0.5)'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging-Level (Standard: INFO)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug-Modus: Speichert Zwischenbilder für jede Verarbeitungsstufe'
    )

    args = parser.parse_args()

    # Setze Logging-Level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Validiere Eingabeverzeichnis
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Eingabeverzeichnis existiert nicht: {input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        logger.error(f"Eingabepfad ist kein Verzeichnis: {input_dir}")
        sys.exit(1)

    # Erstelle Ausgabeverzeichnis falls nötig
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / args.output_csv

    # Validiere Konfidenz-Schwellenwert
    if not 0.0 <= args.confidence <= 1.0:
        logger.error("Konfidenz-Schwelle muss zwischen 0.0 und 1.0 liegen")
        sys.exit(1)

    # Starte Verarbeitung
    logger.info(f"Starte Etiketten-OCR")
    logger.info(f"Eingabeverzeichnis: {input_dir}")
    logger.info(f"Ausgabe-CSV: {output_csv}")
    logger.info(f"Konfidenz-Schwelle: {args.confidence}")
    logger.info(f"Debug-Modus: {'AKTIV' if args.debug else 'INAKTIV'}")

    ocr = EtikettenOCR(
        confidence_threshold=args.confidence,
        debug_mode=args.debug
    )
    ocr.process_directory(input_dir, output_csv)


if __name__ == '__main__':
    # Wenn keine Argumente übergeben wurden, verwende Standardwerte
    if len(sys.argv) == 1:
        logger.info("Keine Argumente angegeben - verwende Standardwerte")
        sys.argv = [
            'etiketten_ocr.py',
            '/home/menas/Downloads/nade/template/',
            '--output-dir', '/home/menas/Downloads/nade/results',
            '--confidence', '0.5',
            '--debug',
            '--log-level', 'DEBUG'
        ]

    main()