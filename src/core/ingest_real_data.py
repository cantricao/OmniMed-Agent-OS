import os
import logging
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# =====================================================================
# ENTERPRISE LOGGING CONFIGURATION
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =====================================================================
# ENVIRONMENT CONFIGURATION
# =====================================================================
load_dotenv()

# [ENTERPRISE FIX] Removed the insecure hardcoded OPENAI_API_KEY hack.
# Langchain may throw a minor warning, but since we rely entirely on local
# HuggingFace embeddings for privacy, external API keys are strictly unnecessary.

# =====================================================================
# REAL DATA INGESTION PIPELINE (Vietnamese Medical Corpus)
# =====================================================================


def download_and_prepare_data(data_path: str):
    """Downloads the ViHealthQA dataset from HuggingFace and saves it locally."""
    logger.info(
        "⏳ [Data Ingestion] Downloading RAG Medical Text data (ViHealthQA) from HuggingFace..."
    )

    # Automatically create directory if it doesn't exist to prevent FileNotFoundError
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    try:
        # Download dataset and save as a CSV file (Bronze Layer)
        ds_dict = load_dataset("tarudesu/ViHealthQA")
        all_splits = [ds_dict[split] for split in ds_dict.keys()]
        ds_combined = concatenate_datasets(all_splits)
        df = ds_combined.to_pandas()

        df.to_csv(data_path, index=False, encoding="utf-8")
        logger.info(f"✅ [Data Ingestion] Data saved successfully to {data_path}")
        return data_path
    except Exception as e:
        logger.critical(
            f"❌ [Data Ingestion] Failed to download dataset.", exc_info=True
        )
        raise


if __name__ == "__main__":
    logger.info("🚀 Starting local RAG ingestion pipeline...")

    # Define local path for CSV database
    data_filepath = "./data/vietnamese_med_corpus/vihealthqa_data.csv"

    # 1. Download data if it doesn't exist locally
    if not os.path.exists(data_filepath):
        download_and_prepare_data(data_filepath)
    else:
        logger.info(
            f"⏭️ [Data Ingestion] Local dataset found at {data_filepath}. Skipping download."
        )

    # 2. Read the CSV data
    logger.info("📖 [Data Ingestion] Loading dataset into memory...")
    df = pd.read_csv(data_filepath)

    # 3. Initialize Local Embedding Model
    logger.info(
        "🧠 [Data Ingestion] Loading embedding model (bkai-foundation-models/vietnamese-bi-encoder)..."
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 4. Process dataframe into Langchain Document objects
    logger.info(
        "🔄 [Data Ingestion] Converting records to Langchain Document format..."
    )
    docs = []
    for index, row in df.iterrows():
        question = row.get("question", row.get("instruction", ""))
        answer = row.get("answer", row.get("output", ""))

        # Skip empty rows to maintain data integrity
        if pd.isna(question) or pd.isna(answer):
            continue

        content = f"Question/Symptom: {question}\nAnalysis/Answer: {answer}"
        docs.append(
            Document(
                page_content=content,
                metadata={"source": "ViHealthQA", "record_id": str(index)},
            )
        )

    logger.info(
        f"💾 [Data Ingestion] Initializing ChromaDB connection for {len(docs)} records..."
    )

    # 5. Initialize ChromaDB connection
    db = Chroma(
        embedding_function=embeddings,
        persist_directory="./data/vietnamese_med_corpus/chroma_db",
        collection_name="vietnamese_ehr_records",
    )

    # 6. Execute Batching to prevent RAM/VRAM overflow during embeddings
    BATCH_SIZE = 1000
    logger.info(
        f"⏳ [Data Ingestion] Starting batch embedding process (Batch size: {BATCH_SIZE})..."
    )
    logger.info(f"☕ This will take some time for {len(docs)} records. Please wait...")

    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Vectorizing Batches"):
        batch = docs[i : i + BATCH_SIZE]
        db.add_documents(documents=batch)

    logger.info(
        "✅ [Data Ingestion] Full dataset ingestion complete! ChromaDB is permanently saved and ready for semantic search."
    )
