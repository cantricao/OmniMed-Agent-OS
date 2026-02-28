import os
from langchain.tools import tool
from docling.document_converter import DocumentConverter

# =====================================================================
# VISION & OCR TOOL SETUP
# =====================================================================
# Utilizing IBM's Docling for advanced document understanding.
# This ensures medical tables (e.g., blood test results) are not scrambled 
# into raw text, but preserved in structured Markdown format.

@tool
def extract_medical_document_ocr(file_path: str) -> str:
    """
    Use this tool to perform OCR and extract structured text, tables, and layouts 
    from scanned medical documents, test results (PDF/Images), or prescriptions.
    
    Args:
        file_path (str): The local system path to the medical document.
        
    Returns:
        str: The extracted document content structured in Markdown format, 
             or an error message if the file is unreadable.
    """
    try:
        print(f"👁️ [Vision Node] Initializing OCR for medical document: '{file_path}'...")
        
        # 1. Validate file existence to prevent pipeline crashes
        if not os.path.exists(file_path):
            error_msg = f"FILE NOT FOUND: The requested document at '{file_path}' does not exist."
            print(f"❌ {error_msg}")
            return error_msg

        # 2. Initialize the Docling converter instance
        # Note: Docling handles complex layouts better than standard Tesseract
        converter = DocumentConverter()
        
        # 3. Execute the OCR and document parsing process
        print(f"⏳ [Vision Node] Parsing tables and layouts. This may take a moment...")
        result = converter.convert(file_path)
        
        # 4. Export to Markdown to preserve table structures for the LLM
        extracted_data = result.document.export_to_markdown()
        
        print("✅ [Vision Node] Document successfully parsed into structured Markdown.")
        
        # Wrap the output in clear boundary markers for the Reasoning Agent
        structured_output = f"--- START OF SCANNED DOCUMENT ({os.path.basename(file_path)}) ---\n\n"
        structured_output += extracted_data
        structured_output += "\n\n--- END OF SCANNED DOCUMENT ---"
        
        return structured_output
        
    except Exception as e:
        # Graceful degradation on OCR failure
        error_msg = f"CRITICAL OCR ERROR processing {file_path}: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

# =====================================================================
# LOCAL TESTING BLOCK (Comment out in production)
# =====================================================================
# if __name__ == "__main__":
#     # Assuming you have a sample lab result in the data folder
#     test_result = extract_medical_document_ocr.invoke("data/images/sample_lab_result.pdf")
#     print(test_result)
