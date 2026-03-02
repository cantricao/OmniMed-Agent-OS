import gradio as gr
import os
import logging
import uuid  # [NEW] Import uuid to generate unique session IDs
from typing import Any, Tuple, Optional
from src.core.main_workflow import omnimed_app

# =====================================================================
# ENTERPRISE LOGGING CONFIGURATION
# =====================================================================
logger = logging.getLogger(__name__)

# =====================================================================
# CORE WEB INTERFACE LOGIC WITH STRICT TYPING & ERROR HANDLING
# =====================================================================
def process_medical_case(
    query: str, 
    patient_id: str, 
    document_file: Any, 
    ref_audio: Optional[str], 
    ref_text: str, 
    llm_model: str
) -> Tuple[str, Optional[str]]:
    """
    Main execution hook for the Gradio UI. 
    Implements programmatic auto-resume for the LangGraph HITL pause to keep UI fluid.
    """
    if document_file is None:
        logger.warning("User attempted to process without uploading a document.")
        gr.Warning("No document uploaded. Please attach a medical record.")
        return "⚠️ **Action Required:** Please upload a document (Image/PDF) to proceed.", None

    doc_path: str = document_file if isinstance(document_file, str) else document_file.name

    state = {
        "doctor_query": query,
        "patient_id": patient_id,
        "document_path": doc_path,
        "prompt_wav_path": ref_audio,
        "prompt_text": ref_text,
        "llm_model_id": llm_model,
    }

    # [NEW] Generate a unique Thread ID for this specific Gradio interaction
    # This ensures that multiple doctors using the app concurrently won't clash memories
    session_id = str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": session_id}}

    try:
        logger.info(f"[Session {session_id}] Initiating analysis for Patient ID: {patient_id}")
        gr.Info("Analyzing medical context... This may take a moment.")
        
        # ---------------------------------------------------------
        # PHASE 1: Run graph until the Human-in-the-Loop Interrupt
        # ---------------------------------------------------------
        logger.info(f"[Session {session_id}] Executing Phase 1 (OCR -> Sanitization -> RAG -> LLM)...")
        # The graph will run and PAUSE exactly before the 'Voice_Alert' node
        paused_state = omnimed_app.invoke(state, config=thread_config)
        
        # Check for explicitly caught errors during Phase 1
        error_msg: Optional[str] = paused_state.get("error_message")
        if error_msg:
            logger.error(f"[Session {session_id}] Workflow error: {error_msg}")
            gr.Warning("Workflow encountered an issue. Check the report panel.")
            return f"### 🚨 System Alert\n\n{error_msg}", None
        
        # Extract the clinical report from the paused state
        report: str = paused_state.get("final_diagnosis", "Failed to generate clinical report.")
        
        # ---------------------------------------------------------
        # PHASE 2: Programmatic Auto-Approve (Resume Graph for Audio)
        # ---------------------------------------------------------
        logger.info(f"[Session {session_id}] Auto-approving HITL pause for UI Demo. Resuming Voice Alert...")
        gr.Info("Clinical reasoning completed. Synthesizing voice alert...")
        
        # Passing 'None' as the state with the same thread_config tells LangGraph to RESUME
        final_state = omnimed_app.invoke(None, config=thread_config)
        
        # Extract the final audio path
        audio_path: Optional[str] = final_state.get("voice_alert_path", None)

        logger.info(f"[Session {session_id}] Analysis and Voice Synthesis completed successfully.")

        if audio_path and os.path.exists(audio_path):
            return report, audio_path
        else:
            return report, None

    except Exception as e:
        logger.critical(f"Catastrophic UI failure during execution: {str(e)}", exc_info=True)
        raise gr.Error(f"Catastrophic System Failure: {str(e)}")

# =====================================================================
# UI/UX DESIGN (GRADIO)
# =====================================================================
with gr.Blocks(title="OmniMed-Agent-OS", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏥 OmniMed-Agent-OS: Multimodal Medical Assistant
        *A fully localized, privacy-first AI system for analyzing medical records locally on constrained hardware.*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input Information")
            doc_input = gr.File(
                label="Upload Document (Receipt/Prescription/X-Ray Image)"
            )
            llm_model_input = gr.Dropdown(
                choices=[
                    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",  # Meta's upgraded Llama 3.1
                    "unsloth/llama-3-8b-Instruct-bnb-4bit",  # Smooth legacy baseline
                    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",  # Alibaba's top-tier reasoning
                    "unsloth/gemma-2-9b-it-bnb-4bit",  # Google's powerhouse
                    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",  # Solid logic
                    "unsloth/Phi-3.5-mini-instruct-bnb-4bit",  # Microsoft's highly efficient model
                ],
                value="unsloth/llama-3-8b-Instruct-bnb-4bit",
                label="🧠 Select Reasoning Model (LLM)",
                info="Choose the local AI model for clinical reasoning (Requires Unsloth support).",
            )
            patient_id_input = gr.Textbox(label="Patient ID", value="BN_001")
            query_input = gr.Textbox(
                label="Doctor's Query / Analysis Command",
                lines=3,
                value="Đây là hóa đơn thanh toán của bệnh nhân. Hãy trích xuất danh sách các mặt hàng/dịch vụ, đơn giá tương ứng và tổng số tiền phải thanh toán từ hình ảnh này.",
            )

            with gr.Accordion("🎙️ Voice Cloning Configuration (Optional)", open=False):
                gr.Markdown(
                    "*Upload or record a short audio clip (3-10s) and provide its exact transcript to clone the voice.*"
                )
                ref_audio_input = gr.Audio(
                    label="Reference Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                    value="data/voice_alerts/sample.wav",
                )
                ref_text_input = gr.Textbox(
                    label="Reference Text (Exact transcript of the audio above)",
                    lines=2,
                    value="Ai đây tức là một kẻ ăn mày vậy. Anh ta chưa kịp quay đi thì đã thấy mấy con chó vàng chạy xồng xộc ra cứ nhảy xổ vào chân anh.",
                )

            submit_btn = gr.Button("🚀 Start AI Analysis", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 📋 Clinical Results & Alerts")
            audio_output = gr.Audio(label="🔊 Voice Alert (VoxCPM)", type="filepath")
            report_output = gr.Markdown(label="Detailed Clinical Report")

    submit_btn.click(
        fn=process_medical_case,
        inputs=[
            query_input,
            patient_id_input,
            doc_input,
            ref_audio_input,
            ref_text_input,
            llm_model_input,
        ],
        outputs=[report_output, audio_output],
    )

if __name__ == "__main__":
    logger.info("🚀 Launching OmniMed Web Interface on local server...")
    demo.launch(share=True, debug=True)