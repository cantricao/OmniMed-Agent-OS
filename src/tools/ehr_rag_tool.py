import os

# [FIXED] Updated to use the dedicated langchain-chroma package
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool

# =====================================================================
# CONFIGURATION
# =====================================================================
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
CHROMA_DB_DIR = "./data/vietnamese_med_corpus/chroma_db"


def get_vietnamese_vector_db() -> Chroma:
    """
    Initializes the embedding model and connects to the local Chroma database.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    db = Chroma(
        collection_name="vietnamese_ehr_records",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    return db


@tool
def search_patient_records(query: str) -> str:
    """
    Use this tool to search for patient medical records (EHR), clinical notes,
    allergies, and medical history from the localized Vietnamese Vector Database.
    """
    try:
        print(f"🔍 [RAG Node] Executing semantic search for query: '{query}'...")

        db = get_vietnamese_vector_db()
        docs = db.similarity_search(query, k=3)

        if not docs:
            return "WARNING: No relevant medical data found in the patient records database."

        context = "\n\n--- RETRIEVED MEDICAL CONTEXT ---\n\n".join(
            [doc.page_content for doc in docs]
        )

        print("✅ [RAG Node] Successfully retrieved relevant medical history.")
        return f"Retrieved Context from Vietnamese EHR Database:\n\n{context}"

    except Exception as e:
        error_msg = f"CRITICAL ERROR retrieving RAG data: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg
