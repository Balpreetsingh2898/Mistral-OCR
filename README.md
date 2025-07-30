# Mistral OCR

A Streamlit-based application for extracting text from PDF and image files using Mistral AI's OCR capabilities.

## Features

- **Single File OCR** (`app.py`): Process individual PDF or image files with immediate results
- **Batch OCR** (`app_b.py`): Process multiple images in bulk using Mistral's batch inference API
- **Structured Output**: Extract text with organized metadata including topics and languages
- **Download Results**: Export OCR results as text files

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API Key**:
   Create a `.env` file in the project directory:
   ```
   MISTRAL_API_KEY=your_api_key_here
   ```

## Usage

### Single File OCR
```bash
streamlit run app.py
```
- Upload a PDF or image file
- Click "Start OCR Process"
- View extracted text and download results

### Batch OCR
```bash
streamlit run app_b.py
```
- Upload multiple image files
- Click "Run Batch OCR"
- Monitor batch job progress
- Download complete results when finished

## Supported Formats

- **Images**: PNG, JPG, JPEG
- **Documents**: PDF

## Requirements

- Python 3.7+
- Mistral AI API key
- Internet connection for API calls

## Author

**Balpreet Singh**  
Email: Balpreetsingh.sn@gmail.com