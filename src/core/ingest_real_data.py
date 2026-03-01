import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from langchain_core.documents import Document 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# [BYPASS TRICK] Declare a fake API Key to suppress LangChain warnings since we run 100% Local AI
os.environ['OPENAI_API_KEY'] = 'fake_key_to_bypass_langchain_warnings'

# =====================================================================
# REAL DATA INGESTION PIPELINE (Vietnamese Medical Corpus)
# =====================================================================

def download_and_prepare_data(data_path: str):
    """Downloads the ViHealthQA dataset from HuggingFace and saves it locally."""
    print("⏳ [Data Ingestion] Downloading RAG Medical Text data (ViHealthQA) from HuggingFace...")
    
    # Automatically create directory if it doesn't exist to prevent FileNotFoundError
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Download dataset and save as a CSV file (Bronze Layer)
    ds_dict = load_dataset("tarudesu/ViHealthQA")
    all_splits = [ds_dict[split] for split in ds_dict.keys()]
    ds_combined = concatenate_datasets(all_splits)
    ds_combined.to_csv(data_path, index=False)
    
    print(f"✅ [Data Ingestion] Successfully downloaded and saved data to: {data_path}")

def ingest_real_vietnamese_medical_data():
    print("🚀 [Data Ingestion] Starting full medical data ingestion into Vector DB...")
    
    data_path = "data/vietnamese_med_corpus/vihealth_qa.csv"
    
    # 1. Check for existing Bronze data; download if missing
    if not os.path.exists(data_path):
        download_and_prepare_data(data_path)
    else:
        print(f"📂 [Data Ingestion] Found existing CSV staging data at {data_path}. Skipping download.")

    # 2. Load Embedding Model
    print("🧠 [Data Ingestion] Loading bkai-foundation-models/vietnamese-bi-encoder...")
    embeddings = HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 3. Read the CSV file into memory
    print(f"📖 [Data Ingestion] Reading full medical dataset from local storage...")
    df = pd.read_csv(data_path)
    
    # 4. Extract columns and compile into Document objects
    print("⏳ [Data Ingestion] Parsing columns and preparing document objects...")
    docs = []
    for index, row in df.iterrows():
        question = row.get('question', row.get('instruction', ''))
        answer = row.get('answer', row.get('output', ''))
        
        # Skip empty rows to maintain data integrity
        if pd.isna(question) or pd.isna(answer):
            continue
            
        content = f"Question/Symptom: {question}\nAnalysis/Answer: {answer}"
        docs.append(Document(page_content=content, metadata={"source": "ViHealthQA", "record_id": str(index)}))
        
    print(f"💾 [Data Ingestion] Initializing ChromaDB connection for {len(docs)} records...")
    
    # 5. Initialize ChromaDB connection
    db = Chroma(
        embedding_function=embeddings,
        persist_directory="./data/vietnamese_med_corpus/chroma_db",
        collection_name="vietnamese_ehr_records"
    )
    
    # 6. Execute Batching to prevent RAM/VRAM overflow during embeddings
    BATCH_SIZE = 1000
    print(f"⏳ [Data Ingestion] Starting batch embedding process (Batch size: {BATCH_SIZE})...")
    print(f"☕ This will take some time for {len(docs)} records. Please wait...")
    
    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Vectorizing Batches"):
        batch = docs[i : i + BATCH_SIZE]
        db.add_documents(documents=batch)
        
    print("✅ [Data Ingestion] Full dataset ingestion complete! ChromaDB is permanently saved and ready for the LangGraph workflow.")

if __name__ == "__main__":
    ingest_real_vietnamese_medical_data()