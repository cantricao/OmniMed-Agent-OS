import sys
import pytest
from unittest.mock import MagicMock, patch

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
# 1. VISION NODE ISOLATION TEST
# =====================================================================
@patch("os.path.exists")
@patch("src.main_workflow.extract_medical_document_ocr")
def test_vision_node_success(mock_ocr_tool, mock_exists, sample_initial_state):
    """
    Tests the Vision node in isolation to ensure it correctly updates
    the state when OCR succeeds.
    """
    # Force the OS to always return True when checking file existence
    mock_exists.return_value = True

    # Setup mock return value for the OCR tool
    mock_ocr_tool.invoke.return_value = "Mocked receipt text."

    result = vision_node(sample_initial_state)

    assert "ocr_extracted_text" in result
    assert result["ocr_extracted_text"] == "Mocked receipt text."


# =====================================================================
# 2. FULL PIPELINE TEST (HITL COMPLIANCE)
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
    # Mock virtual file path to exist
    mock_exists.return_value = True

    # Mock all internal tools within the LangGraph pipeline
    mock_ocr.invoke.return_value = "Mocked OCR text"
    mock_rag.invoke.return_value = "Mocked RAG Context"
    mock_reasoning.invoke.return_value = {
        "final_diagnosis": "Mocked Diagnosis",
        "voice_summary": "Mocked Voice",
    }
    mock_voice.invoke.return_value = "audio.wav"

    # Provide thread configuration required by LangGraph checkpointer
    thread_config = {"configurable": {"thread_id": "ci_test_thread"}}

    # Execute the graph with thread configuration
    result = omnimed_app.invoke(sample_initial_state, config=thread_config)

    # Execution halts at Human-in-the-Loop node; verify initial output
    assert result is not None
    assert "ocr_extracted_text" in result
