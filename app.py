import gradio as gr
import os
import logging
import uuid
from pathlib import Path
from typing import Any, Tuple, Optional
from src.core.main_workflow import omnimed_app
from src.core.config_manager import config

logger = logging.getLogger(__name__)

# Dynamic Pathing
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIO_PATH = BASE_DIR / "data" / "voice_alerts" / "sample.wav"


# =====================================================================
# STEP 1: AI ANALYSIS (Pauses at Doctor Approval)
# =====================================================================
def analyze_medical_case(
    query: str,
    patient_id: str,
    document_file: Any,
    ref_audio: Optional[str],
    ref_text: str,
    llm_model: str,
) -> Tuple[str, Any, str, Any]:
    """Phase 1: Runs OCR, RAG, and LLM reasoning. Pauses before Voice TTS."""
    if document_file is None:
        logger.warning("User attempted to process without uploading a document.")
        gr.Warning("No document uploaded. Please attach a medical record.")
        return (
            "⚠️ **Action Required:** Please upload a document.",
            gr.update(visible=False),
            "",
            gr.update(),
        )

    doc_path: str = (
        document_file if isinstance(document_file, str) else document_file.name
    )

    state = {
        "doctor_query": query,
        "patient_id": patient_id,
        "document_path": doc_path,
        "prompt_wav_path": ref_audio,
        "prompt_text": ref_text,
        "llm_model_id": llm_model,
    }

    session_id = str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": session_id}}

    try:
        logger.info(f"[Session {session_id}] Executing Phase 1 (OCR -> RAG -> LLM)...")
        gr.Info("Analyzing medical context... This may take a moment.")

        paused_state = omnimed_app.invoke(state, config=thread_config)

        error_msg: Optional[str] = paused_state.get("error_message")
        if error_msg:
            return (
                f"### 🚨 System Alert\n\n{error_msg}",
                gr.update(visible=False),
                session_id,
                gr.update(),
            )

        report: str = paused_state.get(
            "final_diagnosis", "Failed to generate clinical report."
        )

        logger.info(
            f"[Session {session_id}] Workflow paused. Waiting for Doctor's approval."
        )
        gr.Info(
            "Analysis complete! Please review the report and approve to generate Voice Alert."
        )

        # Return the report, MAKE THE APPROVE BUTTON VISIBLE, and save the session_id
        return report, gr.update(visible=True), session_id, gr.update(value=None)

    except Exception as e:
        logger.critical(
            f"Catastrophic UI failure during execution: {str(e)}", exc_info=True
        )
        raise gr.Error(f"System Failure: {str(e)}")


# =====================================================================
# STEP 2: DOCTOR APPROVAL (Resumes Graph to generate Voice)
# =====================================================================
def generate_voice_alert(session_id: str) -> str:
    """Phase 2: Resumes the LangGraph thread to synthesize audio."""
    if not session_id:
        raise gr.Error("Session lost. Please re-run the analysis.")

    thread_config = {"configurable": {"thread_id": session_id}}

    try:
        logger.info(
            f"[Session {session_id}] Doctor Approved. Resuming workflow for Voice TTS..."
        )
        gr.Info("Generating Voice Alert... Please wait.")

        final_state = omnimed_app.invoke(None, config=thread_config)
        audio_path: Optional[str] = final_state.get("voice_alert_path", None)

        if audio_path and os.path.exists(audio_path):
            return audio_path
        else:
            raise gr.Error("Voice alert generation failed.")

    except Exception as e:
        logger.critical(f"Voice synthesis failure: {str(e)}", exc_info=True)
        raise gr.Error(f"TTS Failure: {str(e)}")


# =====================================================================
# GRADIO UI LAYOUT
# =====================================================================
with gr.Blocks(title="OmniMed-Agent-OS", theme=gr.themes.Soft()) as demo:
    # State variable to hold the LangGraph thread_id between button clicks
    current_session_id = gr.State("")

    gr.Markdown(
        """
        # 🏥 OmniMed-Agent-OS: Multimodal Medical Assistant
        *A fully localized, privacy-first AI system for analyzing medical records.*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input Information")
            doc_input = gr.File(
                label="Upload Document (Receipt/Prescription/X-Ray Image)"
            )

            models_config = config.get_models()
            llm_model_input = gr.Dropdown(
                choices=models_config.get("available_llms", []),
                value=models_config.get("default_llm"),
                label="🧠 Select Reasoning Model (LLM)",
            )
            patient_id_input = gr.Textbox(label="Patient ID", value="BN_001")
            query_input = gr.Textbox(
                label="Doctor's Query / Analysis Command",
                lines=3,
                value=config.get_prompt("doctor_query"),
            )

            with gr.Accordion("🎙️ Voice Cloning Configuration (Optional)", open=False):
                ref_audio_input = gr.Audio(
                    label="Reference Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                    value=(
                        str(DEFAULT_AUDIO_PATH) if DEFAULT_AUDIO_PATH.exists() else None
                    ),
                )
                ref_text_input = gr.Textbox(
                    label="Reference Text",
                    lines=2,
                    value="Ai đây tức là một kẻ ăn mày vậy...",
                )

            # BUTTON 1: Trigger Phase 1
            submit_btn = gr.Button("🚀 Start AI Analysis", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 📋 Clinical Results & Alerts")
            report_output = gr.Markdown(label="Detailed Clinical Report")

            # BUTTON 2: Hidden by default, appears only after analysis
            approve_btn = gr.Button(
                "👨‍⚕️ Approve Report & Generate Voice Alert",
                visible=False,
                variant="secondary",
            )

            audio_output = gr.Audio(label="🔊 Voice Alert (VoxCPM)", type="filepath")

    # Step 1: Run Analysis -> Output Report & Show Approve Button
    submit_btn.click(
        fn=analyze_medical_case,
        inputs=[
            query_input,
            patient_id_input,
            doc_input,
            ref_audio_input,
            ref_text_input,
            llm_model_input,
        ],
        outputs=[report_output, approve_btn, current_session_id, audio_output],
        concurrency_limit=1,
    )

    # Step 2: Click Approve -> Resume Graph -> Output Audio
    approve_btn.click(
        fn=generate_voice_alert,
        inputs=[current_session_id],
        outputs=[audio_output],
        concurrency_limit=1,
    )

if __name__ == "__main__":
    logger.info("🚀 Launching OmniMed Web Interface on local server...")
    demo.launch(share=True, debug=True)
