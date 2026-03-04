import os
import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool

logger = logging.getLogger(__name__)

# =====================================================================
# CONFIGURATION
# =====================================================================
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
CHROMA_DB_DIR = "./data/vietnamese_med_corpus/chroma_db"

_EMBEDDINGS_CACHE = None
_CHROMA_DB_CACHE = None


def get_vietnamese_vector_db() -> Chroma:
    global _EMBEDDINGS_CACHE, _CHROMA_DB_CACHE

    if _EMBEDDINGS_CACHE is None:
        logger.info(
            "🧠 [RAG Singleton] Loading Embedding Model into VRAM for the first time..."
        )
        _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

    if _CHROMA_DB_CACHE is None:
        logger.info("💾 [RAG Singleton] Connecting to ChromaDB...")
        _CHROMA_DB_CACHE = Chroma(
            collection_name="vietnamese_ehr_records",
            embedding_function=_EMBEDDINGS_CACHE,
            persist_directory=CHROMA_DB_DIR,
        )

    return _CHROMA_DB_CACHE


@tool
def search_patient_records(query: str) -> str:
    """Use this tool to search for patient medical records..."""
    try:
        logger.info(f"🔍 [RAG Node] Executing semantic search for query: '{query}'...")

        db = get_vietnamese_vector_db()
        docs = db.similarity_search(query, k=3)

        if not docs:
            logger.warning(
                "No relevant medical data found in the patient records database."
            )
            return "WARNING: No relevant medical data found in the patient records database."

        context = "\n\n--- RETRIEVED MEDICAL CONTEXT ---\n\n".join(
            [doc.page_content for doc in docs]
        )

        logger.info("✅ [RAG Node] Successfully retrieved relevant medical history.")
        return f"Retrieved Context from Vietnamese EHR Database:\n\n{context}"

    except Exception as e:
        error_msg = f"CRITICAL ERROR retrieving RAG context: {str(e)}"
        logger.error(f"❌ {error_msg}", exc_info=True)
        return error_msg
