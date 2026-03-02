import os
import unsloth
from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END

# Import our custom multimodal tools and local reasoning engine
from src.tools.ocr_vision_tool import extract_medical_document_ocr
from src.tools.ehr_rag_tool import search_patient_records
from src.tools.voice_tts_tool import generate_clinical_voice_alert
from src.core.local_llm import invoke_clinical_reasoning

# =====================================================================
# 1. DEFINE THE GRAPH STATE (Bulletproof Enterprise Standard)
# =====================================================================
# Adding total=False is a lifesaver. It tells Python that not ALL keys 
# need to be present at all times, preventing strict KeyErrors during initialization.
class MedicalState(TypedDict, total=False):
    doctor_query: str
    patient_id: Optional[str]
    document_path: Optional[str]
    llm_model_id: Optional[str]
    prompt_wav_path: Optional[str]
    prompt_text: Optional[str]

    ocr_extracted_text: Optional[str]
    rag_clinical_context: Optional[str]
    final_diagnosis: Optional[str]  # UI Stream
    voice_summary: Optional[str]    # Voice Stream
    voice_alert_path: Optional[str]
    error_message: Optional[str]    # System-wide error tracking

# =====================================================================
# 2. DEFINE THE GRAPH NODES WITH ROBUST ERROR HANDLING
# =====================================================================
def vision_node(state: MedicalState) -> Dict[str, Any]:
    print("\n▶️ [STEP 1] EXECUTING VISION NODE...")
    try:
        doc_path = state.get("document_path")
        if not doc_path or not os.path.exists(doc_path):
            print("⚠️ [Vision Node] No valid document provided. Skipping OCR.")
            return {"ocr_extracted_text": "No document attached."}

        # Safe invocation
        ocr_result = extract_medical_document_ocr.invoke({"file_path": doc_path})
        
        # Ensure we always return a string to the state
        if isinstance(ocr_result, dict):
            # Sometimes Langchain tools return dicts like {"output": "..."}
            ocr_text = ocr_result.get("output", ocr_result.get("text", str(ocr_result)))
        else:
            ocr_text = str(ocr_result)
            
        return {"ocr_extracted_text": ocr_text}
        
    except Exception as e:
        print(f"🚨 [Vision Node Error]: {str(e)}")
        return {"ocr_extracted_text": f"OCR Processing Failed: {str(e)}"}


def rag_node(state: MedicalState) -> Dict[str, Any]:
    print("\n▶️ [STEP 2] EXECUTING RAG NODE...")
    try:
        query = state.get("doctor_query", "")
        rag_result = search_patient_records.invoke({"query": query})
        
        # Ensure we always return a string
        context_str = str(rag_result) if not isinstance(rag_result, dict) else str(rag_result.get("output", rag_result))
        return {"rag_clinical_context": context_str}
        
    except Exception as e:
        print(f"🚨 [RAG Node Error]: {str(e)}")
        return {"rag_clinical_context": "Failed to retrieve medical context."}


def reasoning_node(state: MedicalState) -> Dict[str, Any]:
    print("\n▶️ [STEP 3] EXECUTING CLINICAL REASONING NODE...")
    try:
        selected_model = state.get("llm_model_id", "unsloth/llama-3-8b-Instruct-bnb-4bit")
        print(f"🧠 [Reasoning Node] Using LLM Model: {selected_model}")

        # Invoke the LLM
        llm_result = invoke_clinical_reasoning.invoke(
            {
                "doctor_query": state.get("doctor_query", ""),
                "rag_context": state.get("rag_clinical_context", ""),
                "ocr_text": state.get("ocr_extracted_text", ""),
                "model_name": selected_model,
            }
        )

        # [CRITICAL FIX]: Check if the tool returned a string instead of a dict
        # This prevents the dreaded KeyError / AttributeError when calling .get()
        if isinstance(llm_result, dict):
            final_diag = llm_result.get("final_diagnosis", "Failed to generate UI report.")
            voice_sum = llm_result.get("voice_summary", "Báo cáo đã sẵn sàng.")
        else:
            print("⚠️ [Reasoning Node] LLM returned a raw string instead of a structured dictionary.")
            final_diag = str(llm_result)
            voice_sum = "Hệ thống đã phân tích xong nhưng không thể trích xuất kịch bản giọng nói."

        return {
            "final_diagnosis": final_diag,
            "voice_summary": voice_sum,
        }
        
    except Exception as e:
        print(f"🚨 [Reasoning Node Error]: {str(e)}")
        return {
            "final_diagnosis": f"LLM Inference Failed: {str(e)}",
            "voice_summary": "Đã xảy ra lỗi hệ thống trong quá trình phân tích.",
        }


def voice_node(state: MedicalState) -> Dict[str, Any]:
    print("\n▶️ [STEP 4] EXECUTING VOICE ALERT NODE...")
    try:
        text_to_speak = state.get("voice_summary", "Báo cáo đã sẵn sàng.")

        ref_wav = state.get("prompt_wav_path")
        ref_text = state.get("prompt_text")

        if ref_wav and ref_text:
            print("🎙️ [Voice Node] Voice Cloning Activated using reference audio.")
        else:
            print(f"🔊 [Voice Node] Synthesizing standard audio for TTS: '{text_to_speak}'")

        audio_path = generate_clinical_voice_alert.invoke(
            {
                "clinical_note": text_to_speak,
                "prompt_wav_path": ref_wav,
                "prompt_text": ref_text,
            }
        )
        
        # Extract path if tool returns a dict
        final_audio_path = audio_path.get("output", str(audio_path)) if isinstance(audio_path, dict) else str(audio_path)
        return {"voice_alert_path": final_audio_path}
        
    except Exception as e:
        print(f"🚨 [Voice Node Error]: {str(e)}")
        return {"voice_alert_path": None}


# =====================================================================
# 3. BUILD AND COMPILE THE LANGGRAPH WORKFLOW
# =====================================================================
workflow = StateGraph(MedicalState)

workflow.add_node("Vision_OCR", vision_node)
workflow.add_node("EHR_RAG", rag_node)
workflow.add_node("Clinical_Reasoning", reasoning_node)
workflow.add_node("Voice_Alert", voice_node)

# Define the strict deterministic pipeline
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
    print("\n" + "=" * 50)
    print("🏥 OMNIMED-AGENT-OS: INITIALIZATION COMPLETE")
    print("=" * 50)

    # Mock input for testing the pipeline
    test_state: MedicalState = {
        "doctor_query": "Đây là hóa đơn thanh toán của bệnh nhân. Hãy trích xuất danh sách các mặt hàng/dịch vụ, đơn giá tương ứng và tổng số tiền phải thanh toán từ hình ảnh này.",
        "patient_id": "BN_001",
        "document_path": "data/images/test_receipt.jpg",
        "prompt_wav_path": "data/voice_alerts/sample.wav",
        "prompt_text": "Ai đây tức là một kẻ ăn mày vậy. Anh ta chưa kịp quay đi thì đã thấy mấy con chó vàng chạy xồng xộc ra cứ nhảy xổ vào chân anh.",
    }

    print(f"👨‍⚕️ DOCTOR's QUERY: {test_state.get('doctor_query')}")
    print(f"📄 DOCUMENT ATTACHED: {test_state.get('document_path')}\n")

    # Execute the graph workflow safely
    try:
        final_state = omnimed_app.invoke(test_state)

        print("\n" + "=" * 50)
        print("📋 OMNIMED FINAL CLINICAL REPORT (UI)")
        print("=" * 50)
        print(final_state.get("final_diagnosis"))

        print("\n" + "=" * 50)
        print("🔊 OMNIMED VOICE SUMMARY (TTS)")
        print("=" * 50)
        print(final_state.get("voice_summary"))
        print(f"\n🎙️ AUDIO ALERT STATUS: {final_state.get('voice_alert_path', 'No audio generated.')}")
        
    except Exception as e:
        print(f"\n❌ [Critical Failure] Workflow crashed during execution: {str(e)}")