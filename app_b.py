import streamlit as st
import os
from dotenv import load_dotenv
import base64
import json
import time
from io import BytesIO
from pathlib import Path

# Mistral.ai imports
from mistralai import Mistral

# --- Load environment variables (.env file with MISTRAL_API_KEY=...) ---
load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
if not API_KEY:
    st.error("MISTRAL_API_KEY not found in environment. Please set it in your .env file.")
    st.stop()

# Initialize the Mistral client
client = Mistral(api_key=API_KEY)

# OCR model name
OCR_MODEL = "mistral-ocr-latest"

# Helper function to encode image data as base64
def encode_image_data(image_bytes: bytes) -> str:
    """
    Encodes raw image bytes to a base64-encoded string.
    """
    return base64.b64encode(image_bytes).decode("utf-8")

def create_batch_file(image_urls, output_file: str):
    """
    Creates a JSONL file for the batch inference. 
    Each line includes 'custom_id' and 'body' with the OCR request details.
    """
    with open(output_file, 'w') as file:
        for index, url in enumerate(image_urls):
            entry = {
                "custom_id": str(index),  # Each request gets a custom_id
                "body": {
                    "document": {
                        "type": "image_url",
                        "image_url": url
                    },
                    "include_image_base64": False
                }
            }
            file.write(json.dumps(entry) + '\n')

def main():
    st.title("Mistral Batch OCR Demo")

    st.write(
        """
        This Streamlit app demonstrates how to:
        1. Upload multiple images.
        2. Convert them into a JSONL batch file.
        3. Use Mistral's batch inference to run OCR on all uploaded images in bulk.
        4. Monitor the batch job's progress and download results once complete.
        """
    )

    # --- File upload section ---
    uploaded_files = st.file_uploader(
        "Upload one or more image files (PNG, JPG, JPEG). Then click 'Run Batch OCR'.",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    # We won't create the batch file or run the job until the user clicks a button
    if uploaded_files:
        if st.button("Run Batch OCR"):
            # 1) Convert each image to base64
            st.write("Encoding images in base64...")
            image_urls = []
            for file in uploaded_files:
                img_bytes = file.read()
                base64_img = encode_image_data(img_bytes)
                # Create "data:image/jpeg;base64,..." format
                img_data_url = f"data:image/jpeg;base64,{base64_img}"
                image_urls.append(img_data_url)

            # 2) Create the JSONL batch file
            batch_file_name = "batch_file.jsonl"
            create_batch_file(image_urls, batch_file_name)
            st.write("Created JSONL batch file with OCR requests.")

            # 3) Upload the JSONL file to Mistral
            st.write("Uploading batch file to Mistral...")
            with open(batch_file_name, "rb") as f:
                batch_data = client.files.upload(
                    file={
                        "file_name": batch_file_name,
                        "content": f
                    },
                    purpose="batch"
                )
            st.write(f"File uploaded with ID: {batch_data.id}")

            # 4) Create the batch job
            created_job = client.batch.jobs.create(
                input_files=[batch_data.id],
                model=OCR_MODEL,
                endpoint="/v1/ocr",
                metadata={"job_type": "streamlit_demo"}
            )
            st.write(f"Created batch job with ID: {created_job.id}")

            # 5) Poll for job completion
            st.write("Polling job status. This may take a while for large batches...")
            poll_interval = 2.0  # seconds
            while True:
                retrieved_job = client.batch.jobs.get(job_id=created_job.id)
                total = retrieved_job.total_requests
                succeeded = retrieved_job.succeeded_requests
                failed = retrieved_job.failed_requests

                st.write(
                    f"Status: {retrieved_job.status} | "
                    f"Total: {total} | "
                    f"Succeeded: {succeeded} | "
                    f"Failed: {failed} | "
                    f"Percent Done: {round((succeeded + failed) / total * 100, 2)}%"
                )
                if retrieved_job.status not in ["QUEUED", "RUNNING"]:
                    # Job finished (either COMPLETED or FAILED)
                    break
                time.sleep(poll_interval)
                # Clear output in a normal notebook environment, but in Streamlit
                # we can just keep printing. 
                # If you want to streamline it, you might re-run the script or re-draw.

            # 6) If the job is complete, let the user download the results
            if retrieved_job.status == "COMPLETED":
                st.success("Batch job completed. Downloading the results...")
                # The results are in retrieved_job.output_file
                output_file_id = retrieved_job.output_file
                # Download to memory
                file_content = client.files.download(file_id=output_file_id)
                st.write("Download complete. Here is a preview of the results (first 2 lines):")

                # Show a short preview
                lines = file_content.decode("utf-8").splitlines()
                for line in lines[:2]:
                    st.json(json.loads(line))

                # Provide a download button for the entire results file
                btn = st.download_button(
                    label="Download Full Results",
                    data=file_content,
                    file_name="batch_ocr_results.jsonl",
                    mime="application/json"
                )
            else:
                st.error(f"Batch job ended with status: {retrieved_job.status}")

if __name__ == "__main__":
    main()
