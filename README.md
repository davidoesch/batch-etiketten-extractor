

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

Now for the magic. We will give the terminal your Google Key, and then tell it to process your folder of photos.

**First, give it the key:**

*(Replace YOUR\_KEY\_HERE with the long code you saved in Step 3, keeping the quotation marks).*

* **Windows:** Type set GEMINI\_API\_KEY="YOUR\_KEY\_HERE" and hit Enter.  
* **Mac:** Type export GEMINI\_API\_KEY="YOUR\_KEY\_HERE" and hit Enter.

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

