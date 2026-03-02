import os
import logging
from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END

# Import our custom multimodal tools and local reasoning engine
from src.tools.ocr_vision_tool import extract_medical_document_ocr
from src.tools.ehr_rag_tool import search_patient_records
from src.tools.voice_tts_tool import generate_clinical_voice_alert
from src.core.local_llm import invoke_clinical_reasoning

# =====================================================================
# 0. ENTERPRISE LOGGING CONFIGURATION
# =====================================================================
# Configure the root logger with a standard formatting pattern
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =====================================================================
# 1. DEFINE THE GRAPH STATE
# =====================================================================
class MedicalState(TypedDict, total=False):
    doctor_query: str
    patient_id: Optional[str]
    document_path: Optional[str]
    llm_model_id: Optional[str]
    prompt_wav_path: Optional[str]
    prompt_text: Optional[str]

    ocr_extracted_text: Optional[str]
    rag_clinical_context: Optional[str]
    final_diagnosis: Optional[str]  
    voice_summary: Optional[str]    
    voice_alert_path: Optional[str]
    error_message: Optional[str]    

# =====================================================================
# 2. DEFINE THE GRAPH NODES WITH ROBUST LOGGING
# =====================================================================
def vision_node(state: MedicalState) -> Dict[str, Any]:
    logger.info("▶️ [STEP 1] EXECUTING VISION NODE...")
    try:
        doc_path = state.get("document_path")
        if not doc_path or not os.path.exists(doc_path):
            logger.warning("[Vision Node] No valid document provided. Skipping OCR.")
            return {"ocr_extracted_text": "No document attached."}

        ocr_result = extract_medical_document_ocr.invoke({"file_path": doc_path})
        
        if isinstance(ocr_result, dict):
            ocr_text = ocr_result.get("output", ocr_result.get("text", str(ocr_result)))
        else:
            ocr_text = str(ocr_result)
            
        logger.info("[Vision Node] OCR extraction completed successfully.")
        return {"ocr_extracted_text": ocr_text}
        
    except Exception as e:
        # exc_info=True automatically attaches the stack trace to the log for easy debugging
        logger.error(f"[Vision Node Error]: {str(e)}", exc_info=True)
        return {"ocr_extracted_text": f"OCR Processing Failed: {str(e)}"}


def rag_node(state: MedicalState) -> Dict[str, Any]:
    logger.info("▶️ [STEP 2] EXECUTING RAG NODE...")
    try:
        query = state.get("doctor_query", "")
        rag_result = search_patient_records.invoke({"query": query})
        
        context_str = str(rag_result) if not isinstance(rag_result, dict) else str(rag_result.get("output", rag_result))
        logger.info("[RAG Node] Clinical context retrieved successfully.")
        return {"rag_clinical_context": context_str}
        
    except Exception as e:
        logger.error(f"[RAG Node Error]: {str(e)}", exc_info=True)
        return {"rag_clinical_context": "Failed to retrieve medical context."}


def reasoning_node(state: MedicalState) -> Dict[str, Any]:
    logger.info("▶️ [STEP 3] EXECUTING CLINICAL REASONING NODE...")
    try:
        selected_model = state.get("llm_model_id", "unsloth/llama-3-8b-Instruct-bnb-4bit")
        logger.info(f"[Reasoning Node] Initializing LLM Engine with model: {selected_model}")

        llm_result = invoke_clinical_reasoning.invoke(
            {
                "doctor_query": state.get("doctor_query", ""),
                "rag_context": state.get("rag_clinical_context", ""),
                "ocr_text": state.get("ocr_extracted_text", ""),
                "model_name": selected_model,
            }
        )

        if isinstance(llm_result, dict):
            final_diag = llm_result.get("final_diagnosis", "Failed to generate UI report.")
            voice_sum = llm_result.get("voice_summary", "Báo cáo đã sẵn sàng.")
            logger.info("[Reasoning Node] Clinical diagnosis generated successfully.")
        else:
            logger.warning("[Reasoning Node] LLM returned a raw string instead of a structured dictionary. Fallback applied.")
            final_diag = str(llm_result)
            voice_sum = "Hệ thống đã phân tích xong nhưng không thể trích xuất kịch bản giọng nói."

        return {
            "final_diagnosis": final_diag,
            "voice_summary": voice_sum,
        }
        
    except Exception as e:
        logger.error(f"[Reasoning Node Error]: {str(e)}", exc_info=True)
        return {
            "final_diagnosis": f"LLM Inference Failed: {str(e)}",
            "voice_summary": "Đã xảy ra lỗi hệ thống trong quá trình phân tích.",
        }


def voice_node(state: MedicalState) -> Dict[str, Any]:
    logger.info("▶️ [STEP 4] EXECUTING VOICE ALERT NODE...")
    try:
        text_to_speak = state.get("voice_summary", "Báo cáo đã sẵn sàng.")
        ref_wav = state.get("prompt_wav_path")
        ref_text = state.get("prompt_text")

        if ref_wav and ref_text:
            logger.info("[Voice Node] Voice Cloning Activated using reference audio.")
        else:
            logger.info(f"[Voice Node] Synthesizing standard audio for TTS.")

        audio_path = generate_clinical_voice_alert.invoke(
            {
                "clinical_note": text_to_speak,
                "prompt_wav_path": ref_wav,
                "prompt_text": ref_text,
            }
        )
        
        final_audio_path = audio_path.get("output", str(audio_path)) if isinstance(audio_path, dict) else str(audio_path)
        logger.info("[Voice Node] Audio alert synthesized successfully.")
        return {"voice_alert_path": final_audio_path}
        
    except Exception as e:
        logger.error(f"[Voice Node Error]: {str(e)}", exc_info=True)
        return {"voice_alert_path": None}


# =====================================================================
# 3. BUILD AND COMPILE THE LANGGRAPH WORKFLOW
# =====================================================================
workflow = StateGraph(MedicalState)

workflow.add_node("Vision_OCR", vision_node)
workflow.add_node("EHR_RAG", rag_node)
workflow.add_node("Clinical_Reasoning", reasoning_node)
workflow.add_node("Voice_Alert", voice_node)

workflow.set_entry_point("Vision_OCR")
workflow.add_edge("Vision_OCR", "EHR_RAG")
workflow.add_edge("EHR_RAG", "Clinical_Reasoning")
workflow.add_edge("Clinical_Reasoning", "Voice_Alert")
workflow.add_edge("Voice_Alert", END)

omnimed_app = workflow.compile()

# =====================================================================
# 4. RUNNABLE DEMO / CLI INTERFACE
# =====================================================================
if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("🏥 OMNIMED-AGENT-OS: INITIALIZATION COMPLETE")
    logger.info("=" * 50)

    test_state: MedicalState = {
        "doctor_query": "Đây là hóa đơn thanh toán của bệnh nhân. Hãy trích xuất danh sách các mặt hàng.",
        "patient_id": "BN_001",
        "document_path": "data/images/test_receipt.jpg",
        "prompt_wav_path": "data/voice_alerts/sample.wav",
        "prompt_text": "Ai đây tức là một kẻ ăn mày vậy.",
    }

    logger.info(f"👨‍⚕️ DOCTOR's QUERY: {test_state.get('doctor_query')}")
    logger.info(f"📄 DOCUMENT ATTACHED: {test_state.get('document_path')}")

    try:
        final_state = omnimed_app.invoke(test_state)
        logger.info("=" * 50)
        logger.info("📋 OMNIMED FINAL CLINICAL REPORT (UI)")
        logger.info("=" * 50)
        
        # We use print here intentionally ONLY for the final CLI output display to the user, not for logging
        print(final_state.get("final_diagnosis"))
        print("\n" + "=" * 50)
        print("🔊 OMNIMED VOICE SUMMARY (TTS)")
        print("=" * 50)
        print(final_state.get("voice_summary"))
        
        logger.info(f"🎙️ AUDIO ALERT STATUS: {final_state.get('voice_alert_path', 'No audio generated.')}")
        
    except Exception as e:
        logger.critical(f"❌ [Critical Failure] Workflow crashed during execution: {str(e)}", exc_info=True)