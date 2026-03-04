import os
import logging
from langchain.tools import tool
from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

# =====================================================================
# SINGLETON CACHE
# =====================================================================
_DOC_CONVERTER_CACHE = None


def get_document_converter() -> DocumentConverter:
    global _DOC_CONVERTER_CACHE
    if _DOC_CONVERTER_CACHE is None:
        logger.info(
            "👁️ [Vision Singleton] Initializing Docling OCR Model into Memory..."
        )
        _DOC_CONVERTER_CACHE = DocumentConverter()
    return _DOC_CONVERTER_CACHE


@tool
def extract_medical_document_ocr(file_path: str) -> str:
    """Use this tool to perform OCR and extract structured text..."""
    try:
        logger.info(f"👁️ [Vision Node] Processing medical document: '{file_path}'...")

        if not os.path.exists(file_path):
            error_msg = f"FILE NOT FOUND: The requested document at '{file_path}' does not exist."
            logger.error(f"❌ {error_msg}")
            return error_msg

        converter = get_document_converter()

        logger.info(
            "⏳ [Vision Node] Parsing tables and layouts. This may take a moment..."
        )
        result = converter.convert(file_path)

        extracted_data = result.document.export_to_markdown()

        logger.info(
            "✅ [Vision Node] Document successfully parsed into structured Markdown."
        )

        structured_output = (
            f"--- START OF SCANNED DOCUMENT ({os.path.basename(file_path)}) ---\n\n"
        )
        structured_output += extracted_data
        structured_output += "\n\n--- END OF SCANNED DOCUMENT ---"

        return structured_output

    except Exception as e:
        error_msg = f"CRITICAL OCR ERROR processing {file_path}: {str(e)}"
        logger.error(f"❌ {error_msg}", exc_info=True)
        return error_msg
