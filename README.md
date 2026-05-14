

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

Create a file called `key.txt` in the same folder as the script and paste your Gemini API key as the only line:

```
AIzaSyYourRealKeyGoesHere...
```

`key.txt` is listed in `.gitignore` and will never be committed to version control.

As a fallback the script also accepts the key via the `GEMINI_API_KEY` environment variable (useful for CI or containerised environments).

## Usage

```bash
python gemini_label_processor.py /path/to/input_images 
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

---

## Post-processing Tools

### clean.py — Sanitise and normalise the result CSV

Reads the `_result.csv` produced by the main processor and writes a `_cleaned.csv` with the following fixes applied to every row:

| Rule | What it does |
|------|-------------|
| Strip junk characters | Removes `\|`, `¦`, leading `:` / `;` / `I ` / `! ` from every field |
| Empty → nodata | Blank or single-character garbage becomes `"nodata"` |
| field6 dedup A | Removes the exact `id_number` string from `field6` when present (e.g. `"2-DU-90 11214B"` → `"2-DU-90"` when `id_number` is `"11214B"`) |
| field6 dedup B | Removes the exact `hyphenated_code` string from `field6` when present (e.g. `"2-DU-90 11214B"` → `"11214B"` when `hyphenated_code` is `"2-DU-90"`) |
| OCR fix B→8 | Replaces `B` with `8` in `id_number` (e.g. `"11214B"` → `"112148"`) |
| OCR fix Z→2 | Replaces `Z` with `2` in `id_number` (e.g. `"11214Z"` → `"112142"`) |
| Sort by filename | Rows are sorted numerically by the `filename` column before writing |

**Setup:** edit the `csv_file` path at the top of the script to point to your result CSV, then run:

```bash
python clean.py
```

Output is written next to the input file with `_cleaned` appended to the filename.

---

### find_missing.py — Identify gaps in the filename sequence

Parses the `filename` column (treated as integers) and reports which numbers in the range `[1, MAX]` are absent — useful for spotting images that were skipped or failed.

```bash
# Print missing numbers to stdout
python find_missing.py /path/to/result.csv

# Specify an explicit upper bound (e.g. you know there should be 7023 images)
python find_missing.py /path/to/result.csv 7023

# Save missing numbers to a text file (one per line)
python find_missing.py /path/to/result.csv 7023 --out missing.txt
```

Sample output:

```
Present entries : 6981
Range checked   : 1 to 7023
Missing count   : 42
Missing numbers:
14
203
...
```

---

### plot_missing.py — Visualise gaps as a diagnostic chart

Generates a four-panel PNG showing where gaps are concentrated across the full sequence. Requires `matplotlib`.

```bash
pip install matplotlib

# Auto-detect range from CSV
python plot_missing.py /path/to/result.csv

# Explicit upper bound, custom output path
python plot_missing.py /path/to/result.csv 7023 --out gaps.png
```

Outputs two files:

* `missing_gaps.png` (or the path given with `--out`) — four-panel figure:
  * Presence strip (green = present, red = missing)
  * Gap-size histogram
  * Gap size vs. position scatter
  * Top 20 largest gaps as a ranked bar chart
* `missing_gaps.csv` — machine-readable gap report with columns `gap_start`, `gap_end`, `gap_size`

---

## Limitations

### Hardcoded layout assumptions

Cropping (right 30%) and extraction logic (2 × 3 grid) are tailored to a specific label format. Adjustments are required for other layouts.

### Cloud dependency

Requires active internet connection. No offline fallback available.

### OCR sensitivity

Accuracy depends on image quality, contrast, and label alignment.

## STEP BY STEP INSTRUCTIONS
### ---

**Step 1: Install Python (The "Kitchen")**

Python is the engine that runs our script. Your computer needs it installed to understand the code.

**For Windows:**

1. Go to [python.org/downloads](https://www.python.org/downloads/).  
2. Click the big yellow button that says **Download Python**.  
3. Open the downloaded file to start the installer.  
4. **CRITICAL STEP:** At the very bottom of the first installer window, check the box that says **"Add Python to PATH"**. (If you miss this, the computer won't know where Python is\!).  
5. Click **Install Now** and let it finish.

**For Mac:**

1. Go to [python.org/downloads](https://www.python.org/downloads/).  
2. Click the yellow **Download Python** button.  
3. Open the downloaded file and click "Continue" through the standard installation steps until it finishes.

### ---

**Step 2: Download the Files (The "Ingredients")**

Since you haven't used GitHub before, we will skip the technical way and just download a simple ZIP file.

1. Go to the GitHub page where these files are hosted.  
2. Look for a green button near the top right that says **"\<\> Code"**. Click it.  
3. In the dropdown menu, click **"Download ZIP"**.  
4. Once downloaded, open the ZIP file and extract/unzip the folder.  
5. **Important:** Move that unzipped folder (it might be called batch-etiketten-extractor-main) directly to your **Desktop**. This will make it much easier to find later.

### ---

**Step 3: Get your Google API Key (The "Digital ID Card")**

To use Google's artificial intelligence, you need a personal "Key" so Google knows who is requesting the data.

1. Go to [**Google AI Studio**](https://aistudio.google.com/) and sign in with a Google account.  
2. On the left side of the screen, click **"Get API key"**.  
3. Click the button to **"Create API Key"** (you may be asked to create a new project first, just click yes/create).  
4. **Setting up Billing:** Google gives you a free tier, but to use the API smoothly for thousands of images, you need to add a billing account.  
   * Click on the Google Cloud Console link or billing prompt.  
   * Enter your credit card details (you are only charged if you exceed the massive free limits, and the script is designed to run safely).  
5. Once your key is generated, it will look like a long string of random letters and numbers (e.g., AIzaSy...).  
6. **Copy this key** and paste it into a blank Notepad/TextEdit document for now so you don't lose it. Treat this like a password\!

### ---

**Step 4: Open the Command Line (The "Hacker Window")**

This is where you type commands to talk directly to your computer.

* **On Windows:** Click the Start Menu, type cmd, and click on the app named **Command Prompt**.  
* **On Mac:** Press Command \+ Spacebar to open Spotlight search, type Terminal, and hit Enter.

Now, tell the terminal to look at the folder on your Desktop by typing this and hitting Enter:

* **Mac:** cd \~/Desktop/batch-etiketten-extractor  
* **Windows:** cd Desktop\\batch-etiketten-extractor 
  *(Note: If your folder has a slightly different name, just use that name).*

### ---

**Step 5: Install the Helper Tools**

The script needs two small helper tools to read images and talk to Google. In your terminal, type this exactly and hit Enter:

* **Windows:** pip install pillow google-genai  
* **Mac:** pip3 install pillow google-genai

You will see a bunch of text scroll by as it downloads them. Wait until it stops.

### ---

**Step 6: Run the Script\!**

Now for the magic. We will save your Google Key into a file, and then tell the script to process your folder of photos.

**First, save your key:**

1. Open the project folder on your Desktop (e.g. `batch-etiketten-extractor`).  
2. Create a new plain text file called **`key.txt`** (use Notepad on Windows or TextEdit in plain-text mode on Mac).  
3. Paste the long API key you saved in Step 3 as the only line in the file and save it.  
   The file should contain only: `AIzaSyYourRealKeyGoesHere...`  
4. Make sure the filename is exactly `key.txt` (not `key.txt.txt`).

**Second, run the script:**

You just need to tell the script where your photos are. For example, if your photos are in a folder called 0001-2000 in your Downloads folder:

* **Windows:** python gemini\_label\_processor.py C:\\Users\\YourName\\Downloads\\0001-2000  
* **Mac:** python3 gemini\_label\_processor.py /Users/YourName/Downloads/0001-2000

Hit Enter. The script will wake up, greet you, and start processing files one by one\!

When it finishes, it will automatically create a new folder with all the raw data, and a nice, clean file called 0001-2000\_result.csv right next to your original photo folder.

### ---

**Step 7: Use the Validator (Checking the Work)**

Now that the computer did the hard work, you can review it easily.

1. Go to your Desktop folder (batch-etiketten-extractor).  
2. Find the file named validator.html and **double-click it**. It will open in your web browser (Please use Google Chrome or Microsoft Edge for this).  
3. At the top of the webpage:  
   * Click **"1. Select Original CSV"** and choose the \_result.csv file the script just made.  
   * Click **"2. Select Image Folder"** and choose your original folder of photos. (Your browser might ask "Are you sure you want to upload these files?" Click Yes/Allow. Don't worry, they aren't actually going to the internet, just into your local browser window).  
4. Click **"Start QA Process"**.

You will now see the photo on the left, and the extracted data on the right\! You can use your keyboard to fix any mistakes. When you hit the **Enter** key, it will instantly save your changes directly to the CSV file on your hard drive and load the next photo.

## License

MIT License 

