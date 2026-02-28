import os
import pandas as pd
from tqdm import tqdm # [NEW] Import for progress bar
from langchain_core.documents import Document 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# =====================================================================
# REAL DATA INGESTION PIPELINE (Vietnamese Medical Corpus)
# =====================================================================

def ingest_real_vietnamese_medical_data():
    print("🚀 [Data Ingestion] Starting full medical data ingestion into Vector DB...")
    
    print("🧠 [Data Ingestion] Loading bkai-foundation-models/vietnamese-bi-encoder...")
    embeddings = HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    data_path = "data/vietnamese_med_corpus/vihealth_qa.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Data file not found at {data_path}.")
        print("Please run the HuggingFace dataset download script first!")
        return

    print(f"📖 [Data Ingestion] Reading full medical dataset from: {data_path}...")
    df = pd.read_csv(data_path)
    
    # [REMOVED] sample_df = df.head(500) -> We now process the entire dataframe
    
    print("⏳ [Data Ingestion] Parsing columns and preparing document objects...")
    docs = []
    for index, row in df.iterrows():
        question = row.get('question', row.get('instruction', ''))
        answer = row.get('answer', row.get('output', ''))
        
        if pd.isna(question) or pd.isna(answer):
            continue
            
        # Excellent custom logic for QA mapping
        content = f"Question/Symptom: {question}\nAnalysis/Answer: {answer}"
        docs.append(Document(page_content=content, metadata={"source": "ViHealthQA", "record_id": str(index)}))
        
    print(f"💾 [Data Ingestion] Initializing ChromaDB connection for {len(docs)} records...")
    
    # [NEW] Initialize empty/existing Chroma collection FIRST
    db = Chroma(
        embedding_function=embeddings,
        persist_directory="./data/vietnamese_med_corpus/chroma_db",
        collection_name="vietnamese_ehr_records"
    )
    
    # [NEW] Implement Batching to prevent RAM overflow (Out of Memory)
    BATCH_SIZE = 1000
    print(f"⏳ [Data Ingestion] Starting batch embedding process (Batch size: {BATCH_SIZE})...")
    print("☕ This will take some time for ~40k records. Please wait...")
    
    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Vectorizing Batches"):
        batch = docs[i : i + BATCH_SIZE]
        db.add_documents(documents=batch)
        
    print("✅ [Data Ingestion] Full dataset ingestion complete! ChromaDB is permanently saved and ready.")

if __name__ == "__main__":
    ingest_real_vietnamese_medical_data()