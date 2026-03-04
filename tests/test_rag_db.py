import os
import pytest
import logging
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# =====================================================================
# RAG DATABASE VALIDATION SCRIPT
# =====================================================================


def test_chroma_database():
    logger.info("🔍 [RAG Test] Initializing database connection for validation...")

    # Dynamic Pathing: Calculate root path safely
    base_dir = Path(__file__).resolve().parent.parent
    db_path = base_dir / "data" / "vietnamese_med_corpus" / "chroma_db"
    collection = "vietnamese_ehr_records"

    # [ENTERPRISE FIX] Never use 'return' here.
    # Must explicitly notify the framework that this test is validly SKIPPED.
    if not db_path.exists():
        pytest.skip(
            f"⚠️ ChromaDB not found at {db_path}. Skipping test on CI/CD environment."
        )

    logger.info("🧠 [RAG Test] Loading embedding model...")
    # Force CPU in tests to avoid CUDA errors on CI/CD runners
    embeddings = HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("💾 [RAG Test] Connecting to ChromaDB...")
    db = Chroma(
        persist_directory=str(db_path),
        embedding_function=embeddings,
        collection_name=collection,
    )

    # 3. Verify Total Record Count
    try:
        total_docs = db._collection.count()
        logger.info(
            f"📊 [RAG Test] SUCCESS: Found {total_docs} records in the database!"
        )
    except Exception as e:
        pytest.fail(
            f"Could not retrieve exact count. Database might be corrupted. Details: {e}"
        )

    # 4. Perform a Semantic Search Test
    test_query = "Bệnh nhân bị đau đầu, buồn nôn và chóng mặt kéo dài."
    results = db.similarity_search_with_score(test_query, k=2)

    # Use assert to force test failure if DB returns empty array
    assert len(results) > 0, "❌ [RAG Test] No results found. The database is empty!"

    logger.info("✅ [RAG Test] Search successful! Top 2 retrieved records:")
    for i, (doc, score) in enumerate(results):
        logger.info(f"Result #{i+1} | Distance Score: {score:.4f}")
