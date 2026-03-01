# AVOID PATCHING ERROR (KEYERROR)
import unsloth 

import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

# Import our custom multimodal tools and local reasoning engine
from src.tools.ocr_vision_tool import extract_medical_document_ocr
from src.tools.ehr_rag_tool import search_patient_records
from src.tools.voice_tts_tool import generate_clinical_voice_alert
from src.core.local_llm import invoke_clinical_reasoning

# =====================================================================
# 1. DEFINE THE GRAPH STATE
# =====================================================================
class MedicalState(TypedDict):
    doctor_query: str
    patient_id: Optional[str]
    document_path: Optional[str]
    # [NEW] Voice Cloning parameters
    prompt_wav_path: Optional[str]
    prompt_text: Optional[str]
    
    ocr_extracted_text: Optional[str]
    rag_clinical_context: Optional[str]
    final_diagnosis: Optional[str]    # UI Stream (Detailed, retains English medical terms)
    voice_summary: Optional[str]      # Voice Stream (Strictly pure Vietnamese phonetics)
    voice_alert_path: Optional[str]

# =====================================================================
# 2. DEFINE THE GRAPH NODES
# =====================================================================
def vision_node(state: MedicalState):
    print("\n▶️ [STEP 1] EXECUTING VISION NODE...")
    doc_path = state.get("document_path")
    if not doc_path or not os.path.exists(doc_path):
        print("⚠️ [Vision Node] No valid document provided. Skipping OCR.")
        return {"ocr_extracted_text": "No document attached."}
    
    ocr_result = extract_medical_document_ocr.invoke({"file_path": doc_path})
    return {"ocr_extracted_text": ocr_result}

def rag_node(state: MedicalState):
    print("\n▶️ [STEP 2] EXECUTING RAG NODE...")
    query = state.get("doctor_query", "")
    rag_result = search_patient_records.invoke({"query": query})
    return {"rag_clinical_context": rag_result}

def reasoning_node(state: MedicalState):
    print("\n▶️ [STEP 3] EXECUTING CLINICAL REASONING NODE...")
    
    # Invoke the LLM to generate the Dual-Stream output
    llm_result = invoke_clinical_reasoning.invoke({
        "doctor_query": state.get("doctor_query"),
        "rag_context": state.get("rag_clinical_context"),
        "ocr_text": state.get("ocr_extracted_text")
    })
    
    # Safely extract both streams from the returned dictionary
    return {
        "final_diagnosis": llm_result.get("final_diagnosis", "Failed to generate UI report."),
        "voice_summary": llm_result.get("voice_summary", "Báo cáo đã sẵn sàng.")
    }
    
def voice_node(state: MedicalState):
    print("\n▶️ [STEP 4] EXECUTING VOICE ALERT NODE...")
    text_to_speak = state.get("voice_summary", "Báo cáo đã sẵn sàng.")
    
    # Retrieve voice cloning parameters if they exist
    ref_wav = state.get("prompt_wav_path")
    ref_text = state.get("prompt_text")
    
    if ref_wav and ref_text:
         print("🎙️ [Voice Node] Voice Cloning Activated using reference audio.")
    else:
         print(f"🔊 [Voice Node] Synthesizing standard audio for TTS: '{text_to_speak}'")
    
    # Pass all arguments into the LangChain tool
    audio_path = generate_clinical_voice_alert.invoke({
        "clinical_note": text_to_speak,
        "prompt_wav_path": ref_wav,
        "prompt_text": ref_text
    })
    return {"voice_alert_path": audio_path}


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
    print("\n" + "="*50)
    print("🏥 OMNIMED-AGENT-OS: INITIALIZATION COMPLETE")
    print("="*50)
    
    # Mock input for testing the pipeline
    test_state = {
        "doctor_query": "Đây là hóa đơn thanh toán của bệnh nhân. Hãy trích xuất danh sách các mặt hàng/dịch vụ, đơn giá tương ứng và tổng số tiền phải thanh toán từ hình ảnh này.",
        "patient_id": "BN_001",
        "document_path": "data/images/test_receipt.jpg",
        "prompt_wav_path": "data/voice_alerts/sample.wav",  # Optional: Path to reference audio for voice cloning
        "prompt_text": "Ai đây tức là một kẻ ăn mày vậy. Anh ta chưa kịp quay đi thì đã thấy mấy con chó vàng chạy xồng xộc ra cứ nhảy xổ vào chân anh."
    }
    
    print(f"👨‍⚕️ DOCTOR's QUERY: {test_state['doctor_query']}")
    print(f"📄 DOCUMENT ATTACHED: {test_state['document_path']}\n")
    
    # Execute the graph workflow
    final_state = omnimed_app.invoke(test_state)
    
    print("\n" + "="*50)
    print("📋 OMNIMED FINAL CLINICAL REPORT (UI)")
    print("="*50)
    print(final_state.get("final_diagnosis"))
    
    print("\n" + "="*50)
    print("🔊 OMNIMED VOICE SUMMARY (TTS)")
    print("="*50)
    print(final_state.get("voice_summary"))
    print(f"\n🎙️ AUDIO ALERT STATUS: {final_state.get('voice_alert_path', 'No audio generated.')}")
