import gradio as gr
import os
# Import the pre-configured LangGraph workflow from main_workflow
from src.main_workflow import omnimed_app

# =====================================================================
# CORE WEB INTERFACE LOGIC
# =====================================================================
def process_medical_case(query, patient_id, document_file):
    if document_file is None:
        return "⚠️ Please upload a document (Image/PDF).", None
    
    # Handle file path extraction from Gradio's file object
    doc_path = document_file if isinstance(document_file, str) else document_file.name
    
    # Initialize the input state for the LangGraph pipeline
    state = {
        "doctor_query": query,
        "patient_id": patient_id,
        "document_path": doc_path
    }
    
    try:
        # Trigger the Multi-Agent graph (Vision -> Memory -> Reasoning -> Voice)
        final_state = omnimed_app.invoke(state)
        
        # Retrieve the final outputs from the state dictionary
        report = final_state.get("final_diagnosis", "Failed to generate clinical report.")
        audio_path = final_state.get("voice_alert_path", None)
        
        # Verify if the synthesized audio file exists
        if audio_path and os.path.exists(audio_path):
            return report, audio_path
        else:
            return report, None
            
    except Exception as e:
        return f"❌ Critical System Error occurred: {str(e)}", None

# =====================================================================
# UI/UX DESIGN (GRADIO)
# =====================================================================
# Constructing a clean, professional web interface for medical staff
with gr.Blocks(title="OmniMed-Agent-OS", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏥 OmniMed-Agent-OS: Multimodal Medical Assistant
        *A fully localized, privacy-first AI system for analyzing medical records locally on constrained hardware.*
        """
    )
    
    with gr.Row():
        # LEFT COLUMN: Input Data Requirements
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input Information")
            doc_input = gr.File(label="Upload Document (Receipt/Prescription/X-Ray Image)")
            patient_id_input = gr.Textbox(label="Patient ID", value="BN_001")
            
            # The default query instructs the AI to process billing receipts
            query_input = gr.Textbox(
                label="Doctor's Query / Analysis Command", 
                lines=3, 
                value="Đây là hóa đơn thanh toán của bệnh nhân. Hãy trích xuất danh sách các mặt hàng/dịch vụ, đơn giá tương ứng và tổng số tiền phải thanh toán từ hình ảnh này."
            )
            submit_btn = gr.Button("🚀 Start AI Analysis", variant="primary")
            
        # RIGHT COLUMN: Output Results & Media
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Clinical Results & Alerts")
            audio_output = gr.Audio(label="🔊 Voice Alert (VoxCPM)", type="filepath")
            report_output = gr.Markdown(label="Detailed Clinical Report")
            
    # Connect the execution button to the processing function
    submit_btn.click(
        fn=process_medical_case,
        inputs=[query_input, patient_id_input, doc_input],
        outputs=[report_output, audio_output]
    )

if __name__ == "__main__":
    # The share=True parameter generates a secure public URL via Gradio
    print("🚀 Launching OmniMed Web Interface...")
    demo.launch(share=True, debug=True)
