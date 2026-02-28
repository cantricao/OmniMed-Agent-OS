import torch
from unsloth import FastLanguageModel

# =====================================================================
# LOCAL LLM CORE (Llama-3 8B with Unsloth 4-bit Quantization)
# =====================================================================
# This module serves as the primary reasoning engine for the LangGraph agent.
# It dynamically loads a 4-bit quantized Llama-3 model into the T4 GPU, 
# performs the clinical reasoning, and immediately purges the VRAM.

# Configuration for Unsloth optimized loading
MAX_SEQ_LENGTH = 2048 
DTYPE = None # Auto-detect (usually float16 on T4)
LOAD_IN_4BIT = True 

# Using the pre-quantized Instruction-tuned Llama-3 model
MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"

def invoke_clinical_reasoning(clinical_context: str, user_query: str) -> str:
    """
    Dynamically loads the local LLM, analyzes the medical context (RAG/Vision outputs),
    answers the doctor's query, and frees up the VRAM.
    
    Args:
        clinical_context (str): The combined text from RAG and OCR/Vision tools.
        user_query (str): The specific question or command from the doctor.
        
    Returns:
        str: The generated clinical advice or diagnostic summary.
    """
    try:
        print("🧠 [Reasoning Node] Allocating VRAM for Llama-3 8B (4-bit)...")
        # Ensure VRAM is clean before loading
        torch.cuda.empty_cache()
        
        # 1. Load Model and Tokenizer via Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = MODEL_NAME,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = DTYPE,
            load_in_4bit = LOAD_IN_4BIT,
        )
        
        # Enable Unsloth's 2x faster inference mode
        FastLanguageModel.for_inference(model)
        
        # 2. Construct the Medical Prompt using Chat Templates
        system_prompt = (
            "You are OmniMed, an elite AI medical assistant. "
            "You will be provided with Context (from EHR database) and an Attached Document (OCR text). "
            "CRITICAL RULES: "
            "1. If the Attached Document is a receipt or financial bill, ONLY extract the items, quantities, and prices. DO NOT make clinical diagnoses. "
            "2. If the RAG Context is irrelevant to the Attached Document, STRICTLY IGNORE the RAG Context. Do not invent medical history. "
            "3. YOU MUST OUTPUT THE FINAL REPORT ENTIRELY IN VIETNAMESE (TIẾNG VIỆT). NO ENGLISH ALLOWED."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Medical Data:\n{clinical_context}\n\nDoctor's Query: {user_query}"}
        ]
        
        # Format the prompt for Llama-3 Instruct
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")

        print("🧠 [Reasoning Node] Analyzing data and generating clinical insights...")
        
        # 3. Generate the response
        outputs = model.generate(
            input_ids = inputs, 
            max_new_tokens = 512, 
            use_cache = True,
            temperature = 0.2 # Low temperature for strict, factual medical responses
        )
        
        # Decode the generated tokens (ignoring the input prompt tokens)
        response = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens = True)[0]
        
        # ==========================================================
        # 4. CRITICAL VRAM PURGE
        # ==========================================================
        print("🧹 [Memory Manager] Unloading Llama-3 to free up VRAM for other tools...")
        del model
        del tokenizer
        del inputs
        del outputs
        torch.cuda.empty_cache()
        
        print("✅ [Reasoning Node] Clinical reasoning complete.")
        return response.strip()

    except Exception as e:
        # Fallback cleanup
        torch.cuda.empty_cache()
        error_msg = f"LLM REASONING ERROR: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

# =====================================================================
# LOCAL TESTING BLOCK (Comment out in production)
# =====================================================================
# if __name__ == "__main__":
#     test_context = "Patient shows elevated glucose levels."
#     test_query = "What is the recommended next step?"
#     print(invoke_clinical_reasoning(test_context, test_query))
