import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
import base64
import json
from enum import Enum
import pycountry
from pydantic import BaseModel

# Mistral.ai Python client
from mistralai import Mistral, DocumentURLChunk, ImageURLChunk, TextChunk
from mistralai.models import OCRResponse

# --- Load environment variables (ensure .env has MISTRAL_API_KEY) ---
load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")  
if not API_KEY:
    st.error("MISTRAL_API_KEY not found in environment. Please set it in your .env file.")
    st.stop()

# Initialize the Mistral client
client = Mistral(api_key=API_KEY)

# --- Setup for structured OCR output (sample from your original code) ---
languages = {
    lang.alpha_2: lang.name
    for lang in pycountry.languages
    if hasattr(lang, 'alpha_2')
}

class LanguageMeta(Enum.__class__):
    def __new__(metacls, cls, bases, classdict):
        for code, name in languages.items():
            # Convert name to uppercase with underscores
            classdict[name.upper().replace(' ', '_')] = name
        return super().__new__(metacls, cls, bases, classdict)

class Language(Enum, metaclass=LanguageMeta):
    pass

class StructuredOCR(BaseModel):
    file_name: str
    topics: list[str]
    languages: list[Language]
    ocr_contents: dict

# --- Helper function for structured OCR (images only in this example) ---
def structured_ocr(image_bytes: bytes, image_name: str) -> StructuredOCR:
    """Takes raw bytes of an image, runs OCR, 
       then runs a structured extraction on the results."""
    # Encode file in base64
    encoded_image = base64.b64encode(image_bytes).decode()
    base64_data_url = f"data:image/jpeg;base64,{encoded_image}"

    # Run OCR
    ocr_response = client.ocr.process(
        document=ImageURLChunk(image_url=base64_data_url),
        model="mistral-ocr-latest"
    )

    # Grab markdown from the OCR output
    if len(ocr_response.pages) > 0:
        image_ocr_markdown = ocr_response.pages[0].markdown
    else:
        image_ocr_markdown = ""

    # Use the chat parse method to get structured output
    chat_response = client.chat.parse(
        model="pixtral-12b-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    ImageURLChunk(image_url=base64_data_url),
                    TextChunk(
                        text=(
                            "This is the image's OCR in markdown:\n"
                            f"<BEGIN_IMAGE_OCR>\n{image_ocr_markdown}\n<END_IMAGE_OCR>.\n"
                            "Convert this into a structured JSON response with the OCR contents in a sensible dictionary."
                        )
                    ),
                ],
            },
        ],
        response_format=StructuredOCR,
        temperature=0
    )

    parsed_result: StructuredOCR = chat_response.choices[0].message.parsed

    # If desired, you can replace the 'file_name' field with the actual
    # uploaded file name, etc.:
    parsed_result.file_name = image_name
    return parsed_result

def main():
    st.title("Mistral OCR Demo – Extract and Download Text")

    # Allow user to upload a file (PDF or Image)
    uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
    
    # When the user clicks the "Start OCR Process" button,
    # we will run the OCR only if a file has been uploaded
    if st.button("Start OCR Process"):
        if uploaded_file is not None:
            file_extension = Path(uploaded_file.name).suffix.lower()

            if file_extension == ".pdf":
                st.write("Processing PDF ...")

                # A) Upload the PDF to Mistral
                uploaded_resp = client.files.upload(
                    file={
                        "file_name": Path(uploaded_file.name).stem,
                        "content": uploaded_file.read(),
                    },
                    purpose="ocr",
                )
                # B) Retrieve a signed URL for that file
                signed_url = client.files.get_signed_url(file_id=uploaded_resp.id, expiry=1)

                # C) Process the PDF with OCR
                pdf_response = client.ocr.process(
                    document=DocumentURLChunk(document_url=signed_url.url),
                    model="mistral-ocr-latest",
                    include_image_base64=False
                )

                # Combine OCR text from all pages
                ocr_text = "\n\n".join(page.markdown for page in pdf_response.pages)

                # Display OCR text
                st.subheader("Extracted OCR Text (Markdown)")
                st.text_area("PDF OCR Result:", value=ocr_text, height=300)

                # Download button
                st.download_button(
                    label="Download OCR as Text",
                    data=ocr_text,
                    file_name="ocr_text.txt",
                    mime="text/plain"
                )

            else:
                st.write("Processing Image ...")

                # Read and encode the image
                image_bytes = uploaded_file.read()
                encoded_image = base64.b64encode(image_bytes).decode()
                base64_data_url = f"data:image/jpeg;base64,{encoded_image}"

                # Process with the OCR model
                image_ocr_response = client.ocr.process(
                    document=ImageURLChunk(image_url=base64_data_url),
                    model="mistral-ocr-latest"
                )

                if not image_ocr_response.pages:
                    st.warning("No OCR text found in the image.")
                    return

                # Get the OCR text from first page (or all pages)
                ocr_text = image_ocr_response.pages[0].markdown

                # Display OCR text
                st.subheader("Extracted OCR Text (Markdown)")
                st.text_area("Image OCR Result:", value=ocr_text, height=300)

                # Download button
                st.download_button(
                    label="Download OCR as Text",
                    data=ocr_text,
                    file_name="ocr_text.txt",
                    mime="text/plain"
                )
        else:
            st.warning("Please upload a file before starting the OCR process.")

if __name__ == "__main__":
    main()
