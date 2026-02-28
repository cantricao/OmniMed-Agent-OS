import os
import torch
from unsloth import FastLanguageModel
from langchain.tools import tool

# =====================================================================
# CONFIGURATION
# =====================================================================
MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
DTYPE = None # Auto-detect (bfloat16 for Ampere+, float16 for Tesla T4)
LOAD_IN_4BIT = True

@tool
def invoke_clinical_reasoning(doctor_query: str, rag_context: str, ocr_text: str) -> dict:
    """
    Core reasoning engine using Llama-3 8B to analyze medical data.
    Returns a dictionary containing a detailed UI report and a short voice summary.
    """
    try:
        print("🧠 [Reasoning Node] Allocating VRAM for Llama-3 8B (4-bit)...")
        
        # 1. Load the optimized Unsloth model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(model) # Enable native 2x faster inference
        
        # 2. Construct the Medical Prompt enforcing Dual-Stream output, Diacritic Restoration, and Natural TTS
        system_prompt = (
            "You are OmniMed, an elite AI medical assistant. "
            "You will be provided with Context and an Attached Document (OCR text which may lack Vietnamese accents). "
            "CRITICAL RULES: "
            "1. If it is a receipt, ONLY extract items, quantities, and prices. NO clinical diagnoses. "
            "2. You MUST automatically restore the correct Vietnamese diacritics (khôi phục dấu tiếng Việt) for any unaccented OCR text. "
            "3. ABSOLUTELY NO ENGLISH WORDS in your output, EXCEPT for standard medical/drug names. If info is missing, explain in Vietnamese (e.g., 'Không có thông tin'). "
            "4. DO NOT print my instructions or brackets. "
            "5. YOU MUST OUTPUT EXACTLY TWO SECTIONS using these exact markers:\n"
            "---UI_REPORT---\n"
            "[Your detailed Vietnamese report with restored accents here]\n"
            "---VOICE_SUMMARY---\n"
            "[A short, 1-2 sentence summary in pure Vietnamese with NO English words or numbers, for the voice assistant to read aloud. If you cannot generate a summary, write 'Không thể tạo tóm tắt. Vui lòng xem báo cáo chi tiết.']"
        )
        
        user_message = (
            f"DOCTOR'S QUERY:\n{doctor_query}\n\n"
            f"ATTACHED DOCUMENT (OCR):\n{ocr_text}\n\n"
            f"EHR CONTEXT (RAG):\n{rag_context}\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Apply the chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        print("🧠 [Reasoning Node] Analyzing data and generating clinical insights...")
        
        # 3. Generate the response
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.3, # Low temperature for strict factual medical output
            top_p=0.9
        )
        
        # 4. Decode and extract the generated text
        prompt_length = inputs.shape[1]
        response_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
        
        # 5. Parse the output to separate the UI text and Voice text
        ui_report = response_text
        voice_summary = "Báo cáo đã phân tích xong. Bác sĩ vui lòng xem chi tiết trên màn hình." # Fallback safety
        
        if "---VOICE_SUMMARY---" in response_text:
            parts = response_text.split("---VOICE_SUMMARY---")
            ui_report = parts[0].replace("---UI_REPORT---", "").strip()
            if len(parts) > 1:
                voice_summary = parts[1].strip()
                
        print("🧹 [Memory Manager] Unloading Llama-3 to free up VRAM for other tools...")
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        print("✅ [Reasoning Node] Clinical reasoning complete.")
        
        return {
            "final_diagnosis": ui_report,
            "voice_summary": voice_summary
        }
        
    except Exception as e:
        error_msg = f"CRITICAL REASONING ERROR: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "final_diagnosis": error_msg,
            "voice_summary": "Đã xảy ra lỗi trong quá trình phân tích."
        }