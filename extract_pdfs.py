import pypdf
import os

def extract_text_from_pdf(pdf_path):
    try:
        print(f"--- Extracting text from {os.path.basename(pdf_path)} ---")
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        print(text[:5000]) # Print first 5000 chars to avoid overwhelming output
        print(f"\n--- End of extraction for {os.path.basename(pdf_path)} ---\n")
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

base_dir = "/Users/rafiulhaider/Desktop/IntrotoDS/StreamlitLR/Project_details"
pdf_files = ["CS675-72835-Fall-2025-Project-2-LR.pdf", "Download.pdf"]

for pdf_file in pdf_files:
    extract_text_from_pdf(os.path.join(base_dir, pdf_file))
