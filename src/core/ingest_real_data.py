import os
import pandas as pd
from langchain_core.documents import Document 
# [FIXED] Updated to use the dedicated langchain-chroma package
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# =====================================================================
# REAL DATA INGESTION PIPELINE (Vietnamese Medical Corpus)
# =====================================================================

def ingest_real_vietnamese_medical_data():
    print("🚀 [Data Ingestion] Starting real medical data ingestion into Vector DB...")
    
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

    print(f"📖 [Data Ingestion] Reading real medical dataset from: {data_path}...")
    df = pd.read_csv(data_path)
    
    sample_df = df.head(500) 
    
    docs = []
    for index, row in sample_df.iterrows():
        question = row.get('question', row.get('instruction', ''))
        answer = row.get('answer', row.get('output', ''))
        
        if pd.isna(question) or pd.isna(answer):
            continue
            
        content = f"Question/Symptom: {question}\nAnalysis/Answer: {answer}"
        docs.append(Document(page_content=content, metadata={"source": "ViHealthQA", "record_id": str(index)}))
        
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
