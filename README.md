

# Gemini Label Processor

A privacy-first Python utility for high-throughput extraction of structured data from photographed dot-matrix labels using the Google Gemini Vision API.

## Overview

This tool processes large batches of images containing structured labels. To mitigate exposure of potentially sensitive background information, all images are pre-processed locally. Only the relevant label region is cropped and transmitted to the API.

The extracted metadata is written to structured JSON files, accompanied by a consolidated error log. Original images remain unchanged throughout the workflow.

## Key Features

### Privacy-first pipeline

Crops only the rightmost 30 percent of each image to isolate the label before any external API call.

### Automated orientation correction

Rotates cropped labels 90° counter-clockwise to normalize vertical dot-matrix text into a horizontal layout, improving OCR performance.

### Deterministic grid extraction

Enforces a strict 2 × 3 spatial grid via prompt engineering. Empty fields are preserved as empty strings instead of shifting positions.

### Non-destructive processing

Source images are never modified, moved, or duplicated.

### Resilient API handling

Implements retry logic with exponential backoff for HTTP 429 and 503 responses to ensure stability during long-running batch jobs.

## Requirements

* Python ≥ 3.8
* Gemini API key (Google AI Studio)

## Installation

```bash
git clone https://github.com/davidoesch/batch-etiketten-extractor.git
cd batch-etiketten-extractor
pip install pillow google-genai
```

## Configuration

Set your API key as an environment variable:

### Linux / macOS

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Windows (PowerShell)

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

### Windows (Command Prompt)

```cmd
set GEMINI_API_KEY=your_api_key_here
```

## Usage

```bash
python gemini_label_processor.py /path/to/input_images /path/to/output_metadata
```

## Output Format

For each processed image (e.g. `PB160204.jpg`), a corresponding JSON file is generated:

```json
{
  "id_number": "108479",
  "hyphenated_code": "2-OR-89",
  "field1": "PH FORSCHUNG",
  "field2": "PHARMAZEUTISCHE ENTWICKLUNG",
  "field3": "STERILE DARREICHUNGSFORMEN",
  "field4": "",
  "field5": "D/88",
  "field6": "A1A7",
  "filename": "PB160204"
}
```

If critical fields (`id_number`, `hyphenated_code`) cannot be extracted, the filename is logged in:

```
error_files.txt
```

## Performance Considerations

* Default delay: 4 seconds per request
* Approximate throughput: ~15 requests per minute
* Estimated runtime for 20,000 images: ~22 hours

Using a paid API tier allows reduction or removal of throttling delays.

## Limitations

### Hardcoded layout assumptions

Cropping (right 30%) and extraction logic (2 × 3 grid) are tailored to a specific label format. Adjustments are required for other layouts.

### Cloud dependency

Requires active internet connection. No offline fallback available.

### OCR sensitivity

Accuracy depends on image quality, contrast, and label alignment.

## Roadmap (Optional Enhancements)

* Configurable crop regions via CLI or config file
* Parallel processing with adaptive rate control
* Support for alternative OCR backends (local fallback)
* Direct export to CSV / Excel
* Integration with annotation pipelines (e.g. labeling tools)

## License

MIT License 

