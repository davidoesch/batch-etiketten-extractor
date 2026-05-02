#!/usr/bin/env python3
"""
Etiketten-OCR-Extraktor v3 (Semantic Grid)
Extracts 6 logical fields (L1-L6) from dot-matrix slide labels.
Optimized for specific layout requirements, including grouped bottom-right fields.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# --- Configuration ---
# Set Tesseract path if on Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# PSM 6 assumes a single uniform block of text, good for table-like structures
TESS_CONFIG = r'--oem 3 --psm 6'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class LabelExtractor:
    """
    Specialized class to extract 6 specific fields from slide labels.
    Layout defined as 3 rows x 2 logical columns.
    """

    def __init__(self, confidence_threshold: float = 50.0, debug_mode: bool = False):
        self.confidence_threshold = confidence_threshold
        self.debug_mode = debug_mode

    def _get_roi_and_preprocess(self, image: np.ndarray, debug_path: Optional[Path]) -> np.ndarray:
        """
        1. Crops the right side of the slide image.
        2. Rotates it 90 degrees counter-clockwise.
        3. Applies morphological operations to solidify dot-matrix text.
        """
        h_raw, w_raw = image.shape[:2]

        # 1. Crop & Rotate (Assumes label is in the right ~18%)
        crop_start_x = int(w_raw * 0.82)
        roi = image[0:h_raw, crop_start_x:w_raw]
        roi_rotated = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if debug_path:
             cv2.imwrite(str(debug_path / "01_roi.jpg"), roi_rotated)

        # 2. Preprocessing Sauce (Make dots solid)
        gray = cv2.cvtColor(roi_rotated, cv2.COLOR_BGR2GRAY)
        # Upscale for better OCR recognition
        scale = 3
        gray_up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Binarize (Otsu) -> Text becomes black, BG becomes white
        _, binary = cv2.threshold(gray_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Ensure background is white (Tesseract preference)
        if cv2.countNonZero(binary) < (binary.size * 0.5):
             binary = cv2.bitwise_not(binary)

        # Morphological Erosion (since text is black) to connect dots
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.erode(binary, kernel, iterations=2)

        if debug_path:
            cv2.imwrite(str(debug_path / "02_processed.jpg"), processed)

        return processed

    def _extract_grid_data(self, processed_img: np.ndarray) -> List[Tuple[Dict, Dict]]:
        """
        Core logic: Groups OCR results into a 3x2 grid (3 rows, 2 cells per row).
        Returns a list of 3 tuples: (LeftCellDict, RightCellDict).
        """
        h_img, w_img = processed_img.shape[:2]
        d = pytesseract.image_to_data(processed_img, lang='deu', config=TESS_CONFIG, output_type=Output.DICT)

        valid_boxes = []
        center_zone_start = w_img * 0.45
        center_zone_end = w_img * 0.65

        # 1. Initial Filtering and Box Collection
        for i in range(len(d['text'])):
            text = d['text'][i].strip()
            conf = int(d['conf'][i])
            if conf < 30 or not text: continue

            x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
            x_center = x + w / 2
            y_center = y + h / 2

            # Filter separator noise in the middle zone
            if center_zone_start < x_center < center_zone_end:
                if text in ['|', ':', ';', '!', 'I', '1', '.', ',', '/']:
                    continue

            valid_boxes.append({'text': text, 'conf': conf, 'xc': x_center, 'yc': y_center, 'x': x, 'w': w})

        if not valid_boxes:
             return []

        # 2. Cluster into ROWS based on Y-center coordinates
        # Sort boxes vertically to help clustering
        valid_boxes.sort(key=lambda b: b['yc'])
        rows = []
        current_row = []
        if valid_boxes:
            current_row.append(valid_boxes[0])

            for i in range(1, len(valid_boxes)):
                # If y-center is close to previous box's y-center (within 25px tolerance on upscaled img)
                if abs(valid_boxes[i]['yc'] - valid_boxes[i-1]['yc']) < 35:
                    current_row.append(valid_boxes[i])
                else:
                    # New row started
                    rows.append(current_row)
                    current_row = [valid_boxes[i]]
            rows.append(current_row)

        # We expect exactly 3 rows. Take top 3 if more, pad if fewer.
        rows = rows[:3]

        grid_rows = []

        # 3. Process each ROW to split into LEFT and RIGHT cells
        for row_boxes in rows:
            # Sort horizontally
            row_boxes.sort(key=lambda b: b['xc'])

            # Find the largest horizontal gap between adjacent words in this row
            max_gap = 0
            split_index = 0

            # Default split if only one item exists or no clear gap found
            if len(row_boxes) > 0:
                 split_index = len(row_boxes) # Default: everything is "Left"

            for i in range(len(row_boxes) - 1):
                # Gap between end of current box and start of next box
                gap = row_boxes[i+1]['x'] - (row_boxes[i]['x'] + row_boxes[i]['w'])
                # We only care about gaps roughly in the middle of the image
                if gap > max_gap and (w_img * 0.3 < row_boxes[i]['xc'] < w_img * 0.8):
                    max_gap = gap
                    split_index = i + 1

            # Special case handles single-word rows that are clearly on the right side
            if len(row_boxes) == 1 and row_boxes[0]['xc'] > w_img * 0.6:
                 split_index = 0

            left_boxes = row_boxes[:split_index]
            right_boxes = row_boxes[split_index:]

            # Helper to aggregate boxes into a single cell result
            def aggregate_cell(boxes):
                if not boxes: return {'text': '', 'conf': 0.0}
                full_text = " ".join([b['text'] for b in boxes])
                avg_conf = sum([b['conf'] for b in boxes]) / len(boxes)
                return {'text': full_text, 'conf': avg_conf}

            grid_rows.append((aggregate_cell(left_boxes), aggregate_cell(right_boxes)))

        # Pad to ensure 3 rows exist
        while len(grid_rows) < 3:
             grid_rows.append(({'text': '', 'conf': 0.0}, {'text': '', 'conf': 0.0}))

        return grid_rows

    def process_image(self, image_path: Path) -> Optional[Dict[str, Any]]:
        try:
            debug_path = None
            if self.debug_mode:
                debug_path = image_path.parent / f"debug_{image_path.stem}"
                debug_path.mkdir(exist_ok=True)

            image = cv2.imread(str(image_path))
            if image is None:
                 logger.error(f"Could not read image: {image_path}")
                 return None

            # Step 1: Get optimized image part
            processed_img = self._get_roi_and_preprocess(image, debug_path)

            # Step 2: Extract data into 3x2 grid
            # grid_rows is a list of tuples [(L1, R1), (L2, R2), (L3, R3)]
            grid_rows = self._extract_grid_data(processed_img)

            # Step 3: Map grid to flat L1-L6 structure
            fields = {}
            total_conf = 0
            field_count = 0

            # Row 1 -> L1, L2
            fields['L1_text'] = grid_rows[0][0]['text']
            fields['L1_conf'] = grid_rows[0][0]['conf']
            fields['L2_text'] = grid_rows[0][1]['text']
            fields['L2_conf'] = grid_rows[0][1]['conf']

            # Row 2 -> L3, L4
            fields['L3_text'] = grid_rows[1][0]['text']
            fields['L3_conf'] = grid_rows[1][0]['conf']
            fields['L4_text'] = grid_rows[1][1]['text']
            fields['L4_conf'] = grid_rows[1][1]['conf']

            # Row 3 -> L5, L6 (L6 will contain grouped text if present)
            fields['L5_text'] = grid_rows[2][0]['text']
            fields['L5_conf'] = grid_rows[2][0]['conf']
            fields['L6_text'] = grid_rows[2][1]['text']
            fields['L6_conf'] = grid_rows[2][1]['conf']

            # Calculate average confidence only for non-empty fields
            confs = [v for k, v in fields.items() if '_conf' in k and fields[k.replace('_conf', '_text')]]
            avg_conf = sum(confs) / len(confs) if confs else 0.0

            result = {
                'filename': image_path.name,
                **fields,
                'avg_confidence': avg_conf,
                'manual_check': "CHECK" if avg_conf < self.confidence_threshold else ""
            }
            return result

        except Exception as e:
            logger.exception(f"Failed to process {image_path.name}")
            return None

    def process_directory(self, input_dir: Path, output_csv: Path):
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG'}
        image_files = sorted([f for f in input_dir.iterdir() if f.suffix in image_extensions])

        logger.info(f"Found {len(image_files)} images in {input_dir}")
        results = []

        for img_path in image_files:
            logger.info(f"Processing {img_path.name}...")
            res = self.process_image(img_path)
            if res: results.append(res)

        self._write_csv(results, output_csv)

    def _write_csv(self, results: List[Dict], output_csv: Path):
        if not results:
             logger.warning("No results to write to CSV.")
             return

        # Define order: Filename, L1-L6 texts, L1-L6 scores, Check
        headers = ['filename',
                   'L1_text', 'L2_text', 'L3_text', 'L4_text', 'L5_text', 'L6_text',
                   'L1_conf', 'L2_conf', 'L3_conf', 'L4_conf', 'L5_conf', 'L6_conf',
                   'avg_confidence', 'manual_check']

        try:
            with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(results)
            logger.info(f"Successfully wrote results to: {output_csv}")
        except IOError as e:
             logger.error(f"Could not write CSV file: {e}")

def main():
    parser = argparse.ArgumentParser(description='OCR Extractor for Dot-Matrix Slide Labels (L1-L6 format)')
    parser.add_argument('input_dir', type=str, help='Directory containing images')
    parser.add_argument('--confidence', type=float, default=50.0, help='Average confidence threshold for manual check flag (default: 50.0)')
    parser.add_argument('--debug', action='store_true', help='Save intermediate processing images')

    # Optional: Hardcode paths for quick testing in IDEs
    # sys.argv.extend([r"C:\Path\To\Your\Images", "--debug"])

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    if not input_path.is_dir():
         logger.error(f"Input directory does not exist: {input_path}")
         sys.exit(1)

    output_csv = input_path / 'label_data_L1-L6.csv'

    extractor = LabelExtractor(confidence_threshold=args.confidence, debug_mode=args.debug)
    extractor.process_directory(input_path, output_csv)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Bitte Argumente übergeben oder Hardcode Pfad hier aktivieren.")
        sys.argv.append('/home/menas/Downloads/nade/best/')
        sys.argv.append('--debug')
    main()