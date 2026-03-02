import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# =====================================================================
# RAG DATABASE VALIDATION SCRIPT
# Used to verify vector ingestion and test semantic search accuracy
# =====================================================================


def test_chroma_database():
    print("🔍 [RAG Test] Initializing database connection for validation...")

    db_path = "./data/vietnamese_med_corpus/chroma_db"
    collection = "vietnamese_ehr_records"

    if not os.path.exists(db_path):
        print(f"❌ [RAG Test] ERROR: Database directory not found at {db_path}")
        return

    # 1. Load the exact same embedding model used during ingestion
    print(
        "🧠 [RAG Test] Loading embedding model (bkai-foundation-models/vietnamese-bi-encoder)..."
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 2. Connect to the persisted database
    print("💾 [RAG Test] Connecting to ChromaDB...")
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=collection,
    )

    # 3. Verify Total Record Count
    try:
        # Accessing the underlying collection directly to get the count
        total_docs = db._collection.count()
        print(f"\n📊 [RAG Test] SUCCESS: Found {total_docs} records in the database!")
    except Exception as e:
        print(f"\n⚠️ [RAG Test] Could not retrieve exact count. Details: {e}")

    # 4. Perform a Semantic Search Test
    test_query = "Bệnh nhân bị đau đầu, buồn nôn và chóng mặt kéo dài."
    print(f"\n🩺 [RAG Test] Running sample vector search for: '{test_query}'")

    # Retrieve top 2 most relevant chunks based on vector distance
    results = db.similarity_search_with_score(test_query, k=2)

    if not results:
        print(
            "❌ [RAG Test] No results found. The database might be empty or corrupted."
        )
        return

    print("\n✅ [RAG Test] Search successful! Top 2 retrieved records:")
    print("=" * 60)
    for i, (doc, score) in enumerate(results):
        # A lower score typically means shorter distance/higher similarity
        print(f"Result #{i+1} | Distance Score: {score:.4f}")
        print(f"Metadata: {doc.metadata}")
        # Print a snippet of the content to verify it's not garbage data
        print(f"Content Snippet: {doc.page_content[:150]}...")
        print("-" * 60)


if __name__ == "__main__":
    test_chroma_database()
