import os
import pandas as pd
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# =====================================================================
# REAL DATA INGESTION PIPELINE (Vietnamese Medical Corpus)
# =====================================================================

def ingest_real_vietnamese_medical_data():
    print("🚀 [Data Ingestion] Starting real medical data ingestion into Vector DB...")
    
    # 1. Initialize the official BKAI Embedding Model (Vietnamese native model)
    print("🧠 [Data Ingestion] Loading bkai-foundation-models/vietnamese-bi-encoder...")
    embeddings = HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder",
        model_kwargs={'device': 'cuda'}, # Force GPU for faster embedding
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 2. Verify and read the downloaded real dataset (ViHealthQA or equivalent)
    # Note: Ensure this CSV file exists in the directory after the download step
    data_path = "data/vietnamese_med_corpus/vihealth_qa.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Data file not found at {data_path}.")
        print("Please run the HuggingFace dataset download script first!")
        return

    print(f"📖 [Data Ingestion] Reading real medical dataset from: {data_path}...")
    df = pd.read_csv(data_path)
    
    # Take the first 500 records for testing (can be increased later, 
    # but 500 is optimal to test RAG without extreme GPU wait times)
    sample_df = df.head(500) 
    
    docs = []
    for index, row in sample_df.iterrows():
        # Handle column names based on dataset format 
        # (usually 'question'/'answer' or 'instruction'/'output')
        question = row.get('question', row.get('instruction', ''))
        answer = row.get('answer', row.get('output', ''))
        
        if pd.isna(question) or pd.isna(answer):
            continue
            
        # Combine into a single comprehensive medical document
        # The content remains in Vietnamese for the AI to read, 
        # but the structure is strictly maintained
        content = f"Question/Symptom: {question}\nAnalysis/Answer: {answer}"
        
        # Add metadata for tracking the source
        docs.append(Document(page_content=content, metadata={"source": "ViHealthQA", "record_id": str(index)}))
        
    # 3. Inject the formatted documents into ChromaDB
    print(f"💾 [Data Ingestion] Persisting {len(docs)} medical records into ChromaDB...")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./data/vietnamese_med_corpus/chroma_db",
        collection_name="vietnamese_ehr_records"
    )
    
    print("✅ [Data Ingestion] Real data ingestion complete! ChromaDB is ready for the RAG Node.")

if __name__ == "__main__":
    ingest_real_vietnamese_medical_data()
