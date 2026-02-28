import gradio as gr
import os
from src.main_workflow import omnimed_app

# =====================================================================
# CORE WEB INTERFACE LOGIC
# =====================================================================
def process_medical_case(query, patient_id, document_file, ref_audio, ref_text):
    if document_file is None:
        return "⚠️ Please upload a document (Image/PDF).", None
    
    doc_path = document_file if isinstance(document_file, str) else document_file.name
    
    # [NEW] Inject voice cloning parameters into the graph state
    state = {
        "doctor_query": query,
        "patient_id": patient_id,
        "document_path": doc_path,
        "prompt_wav_path": ref_audio,
        "prompt_text": ref_text
    }
    
    try:
        final_state = omnimed_app.invoke(state)
        report = final_state.get("final_diagnosis", "Failed to generate clinical report.")
        audio_path = final_state.get("voice_alert_path", None)
        
        if audio_path and os.path.exists(audio_path):
            return report, audio_path
        else:
            return report, None
            
    except Exception as e:
        return f"❌ Critical System Error occurred: {str(e)}", None

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
            doc_input = gr.File(label="Upload Document (Receipt/Prescription/X-Ray Image)")
            patient_id_input = gr.Textbox(label="Patient ID", value="BN_001")
            query_input = gr.Textbox(
                label="Doctor's Query / Analysis Command", 
                lines=3, 
                value="Đây là hóa đơn thanh toán của bệnh nhân. Hãy trích xuất danh sách các mặt hàng/dịch vụ, đơn giá tương ứng và tổng số tiền phải thanh toán từ hình ảnh này."
            )
            
            # [NEW] Voice Cloning Section
            with gr.Accordion("🎙️ Voice Cloning Configuration (Optional)", open=False):
                gr.Markdown("*Upload or record a short audio clip (3-10s) and provide its exact transcript to clone the voice.*")
                ref_audio_input = gr.Audio(label="Reference Audio", type="filepath", sources=["upload", "microphone"])
                ref_text_input = gr.Textbox(label="Reference Text (Exact transcript of the audio above)", lines=2)
                
            submit_btn = gr.Button("🚀 Start AI Analysis", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Clinical Results & Alerts")
            audio_output = gr.Audio(label="🔊 Voice Alert (VoxCPM)", type="filepath")
            report_output = gr.Markdown(label="Detailed Clinical Report")
            
    # [NEW] Add ref_audio and ref_text to the inputs list
    submit_btn.click(
        fn=process_medical_case,
        inputs=[query_input, patient_id_input, doc_input, ref_audio_input, ref_text_input],
        outputs=[report_output, audio_output]
    )

if __name__ == "__main__":
    print("🚀 Launching OmniMed Web Interface...")
    demo.launch(share=True, debug=True)