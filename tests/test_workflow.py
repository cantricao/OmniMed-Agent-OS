import sys
from unittest.mock import MagicMock

# =====================================================================
# ENTERPRISE MOCKING: FAKE HEAVY ML LIBRARIES FOR CI
# We mock 'unsloth' and 'torch' BEFORE importing our local modules
# to prevent ModuleNotFoundError on CPU-only CI environments.
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
from src.main_workflow import omnimed_app, MedicalState
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# =====================================================================
# UNIT TESTS FOR LANGGRAPH WORKFLOW (MOCKED AI COMPONENTS)
# =====================================================================


@pytest.fixture
def sample_initial_state() -> MedicalState:
    """Provides a standard baseline state for testing."""
    return {
        "doctor_query": "Extract billing details.",
        "patient_id": "TEST_001",
        "document_path": "dummy_path.jpg",  # Path doesn't need to exist because we mock OCR
        "llm_model_id": "mock_model",
        "prompt_wav_path": None,
        "prompt_text": None,
    }


@patch("src.main_workflow.extract_medical_document_ocr")
def test_vision_node_success(mock_ocr_tool, sample_initial_state):
    """
    Tests the Vision node in isolation to ensure it correctly updates
    the state when OCR succeeds.
    """
    # 1. Arrange: Setup the mock to return a fake OCR dictionary
    mock_ocr_tool.invoke.return_value = {"output": "Mocked receipt text."}

    # 2. Act: Import the node function directly and run it
    from src.main_workflow import vision_node

    result = vision_node(sample_initial_state)

    # 3. Assert: Verify the state was updated correctly
    assert "ocr_extracted_text" in result
    assert result["ocr_extracted_text"] == "Mocked receipt text."
    mock_ocr_tool.invoke.assert_called_once()


@patch("src.main_workflow.generate_clinical_voice_alert")
@patch("src.main_workflow.invoke_clinical_reasoning")
@patch("src.main_workflow.search_patient_records")
@patch("src.main_workflow.extract_medical_document_ocr")
@patch("os.path.exists")
def test_full_langgraph_pipeline_execution(
    mock_exists, mock_ocr, mock_rag, mock_llm, mock_voice, sample_initial_state
):
    """
    Tests the complete StateGraph execution from start to finish.
    All heavy ML components are mocked to ensure rapid, deterministic testing.
    """
    # 1. Arrange: Mock system checks and all tool responses
    mock_exists.invoke.return_value = True
    mock_ocr.invoke.return_value = {"output": "Extracted text"}
    mock_rag.invoke.return_value = {"output": "Clinical context"}
    mock_llm.invoke.return_value = {
        "final_diagnosis": "Patient has a cold.",
        "voice_summary": "Bệnh nhân bị cảm.",
    }
    mock_voice.return_value = {"output": "fake_audio.wav"}

    # 2. Act: Invoke the compiled LangGraph application
    final_state = omnimed_app.invoke(sample_initial_state)

    # 3. Assert: Verify the graph traversed all nodes and populated the final state
    assert final_state.get("ocr_extracted_text") == "Extracted text"
    assert final_state.get("rag_clinical_context") == "Clinical context"
    assert final_state.get("final_diagnosis") == "Patient has a cold."
    assert final_state.get("voice_summary") == "Bệnh nhân bị cảm."
    assert final_state.get("voice_alert_path") == "fake_audio.wav"

    # Verify that every node in our pipeline was triggered exactly once
    mock_ocr.assert_called_once()
    mock_rag.assert_called_once()
    mock_llm.assert_called_once()
    mock_voice.assert_called_once()
