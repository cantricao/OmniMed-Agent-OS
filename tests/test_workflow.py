import sys
from unittest.mock import MagicMock, patch
import pytest

# =====================================================================
# ENTERPRISE MOCKING: FAKE HEAVY ML LIBRARIES FOR CI
# We mock heavy libraries BEFORE importing local modules
# to prevent ModuleNotFoundError and C++ library errors on CI environments.
# =====================================================================
mock_modules = [
    "unsloth",
    "unsloth.models",
    "triton",
    "xformers",
    "bitsandbytes",
    "docling",
    "docling.document_converter",
    "transformers",
    "torch",
    "torchaudio",
    "voxcpm",
]

for module in mock_modules:
    sys.modules[module] = MagicMock()

# Now it's safe to import your actual code
from src.main_workflow import omnimed_app, vision_node


@pytest.fixture
def sample_initial_state():
    return {
        "doctor_query": "Extract billing details.",
        "document_path": "dummy_path.jpg",
        "llm_model_id": "mock_model",
        "patient_id": "TEST_001",
    }


# =====================================================================
# 1. FIX CHO TEST VISION NODE
# =====================================================================
@patch("os.path.exists")
@patch("src.main_workflow.extract_medical_document_ocr")
def test_vision_node_success(mock_ocr_tool, mock_exists, sample_initial_state):
    """
    Tests the Vision node in isolation to ensure it correctly updates
    the state when OCR succeeds.
    """
    mock_exists.return_value = True

    mock_ocr_tool.invoke.return_value = "Mocked receipt text."

    result = vision_node(sample_initial_state)

    assert "ocr_extracted_text" in result
    assert result["ocr_extracted_text"] == "Mocked receipt text."


# =====================================================================
# 2. FIX CHO TEST FULL PIPELINE (HITL COMPLIANCE)
# =====================================================================
@patch("os.path.exists")
@patch("src.main_workflow.extract_medical_document_ocr")
@patch("src.main_workflow.search_patient_records")
@patch("src.main_workflow.invoke_clinical_reasoning")
@patch("src.main_workflow.generate_clinical_voice_alert")
def test_full_langgraph_pipeline_execution(
    mock_voice, mock_reasoning, mock_rag, mock_ocr, mock_exists, sample_initial_state
):
    """
    Tests the entire LangGraph orchestration with Human-in-the-Loop memory.
    """
    mock_exists.return_value = True

    mock_ocr.invoke.return_value = "Mocked OCR text"
    mock_rag.invoke.return_value = "Mocked RAG Context"
    mock_reasoning.invoke.return_value = {
        "final_diagnosis": "Mocked Diagnosis",
        "voice_summary": "Mocked Voice",
    }
    mock_voice.invoke.return_value = "audio.wav"

    thread_config = {"configurable": {"thread_id": "ci_test_thread"}}

    result = omnimed_app.invoke(sample_initial_state, config=thread_config)

    assert result is not None
    assert "ocr_extracted_text" in result
