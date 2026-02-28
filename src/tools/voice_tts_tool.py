import os
import torch
from langchain.tools import tool
import soundfile as sf
from voxcpm import VoxCPM

# =====================================================================
# MODEL DOWNLOAD & SETUP CONFIGURATION
# =====================================================================
# Define the HuggingFace repository ID for VoxCPM
# Change this if you are using a specific or fine-tuned variant
HF_REPO_ID = "JayLL13/VoxCPM-1.5-VN"


# =====================================================================
# DIRECT TTS TOOL (VoxCPM Local Inference with Dynamic VRAM Management)
# =====================================================================

@tool
def generate_clinical_voice_alert(clinical_note: str, prompt_wav_path: str = None, prompt_text: str = None) -> str:
    """
    Use this tool STRICTLY when you need to broadcast a clinical warning, 
    drug interaction alert, or medical summary to the doctor via voice.
    
    Args:
        clinical_note (str): The medical text (Vietnamese) to be synthesized into speech.
        
    Returns:
        str: A confirmation message containing the absolute path to the generated audio file.
    """
    try:
        print(f"🎙️ [Voice Node] Initiating local TTS synthesis...")
        
        # 0. Ensure the model is downloaded and get the correct path
        # checkpoint_path = ensure_model_downloaded()
        
        print("🧹 [Memory Manager] Clearing VRAM to prepare for VoxCPM...")
        # 1. Forcefully release unreferenced GPU memory from previous nodes (Vision/LLM)
        torch.cuda.empty_cache() 
        
        # 2. Dynamically load the VoxCPM model into GPU ONLY when called
        print("📥 [Voice Node] Loading VoxCPM model into GPU. This may take a moment...")
        
        # Instantiate the model from the local downloaded path
        current_model = VoxCPM.from_pretrained(HF_REPO_ID)
        
        # Ensure the output directory exists
        output_dir = "data/voice_alerts"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "clinical_alert.wav")
        
        # 3. Execute the inference to generate speech
        print(f"🔊 [Voice Node] Synthesizing audio for: '{clinical_note[:50]}...'")
        
        wav = current_model.generate(
            text= clinical_note,
            prompt_wav_path=prompt_wav_path if prompt_wav_path else None,      # optional: path to a prompt speech for voice cloning
            prompt_text=prompt_text if prompt_text else None,          # optional: reference text
            cfg_value=2.0,             # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
            inference_timesteps=10,   # LocDiT inference timesteps, higher for better result, lower for fast speed
            normalize=False,           # enable external TN tool, but will disable native raw text support
            denoise=False,             # enable external Denoise tool, but it may cause some distortion and restrict the sampling rate to 16kHz
            retry_badcase=True,        # enable retrying mode for some bad cases (unstoppable)
            retry_badcase_max_times=3,  # maximum retrying times
            retry_badcase_ratio_threshold=6.0, # maximum length restriction for bad case detection (simple but effective), it could be adjusted for slow pace speech
        )

        sf.write(output_file, wav, current_model.tts_model.sample_rate)
        
        # ==========================================================
        # 4. CRITICAL STEP: Purge the model from VRAM instantly
        # ==========================================================
        print("🧹 [Memory Manager] Unloading VoxCPM and freeing up VRAM...")
        del current_model
        del wav
        torch.cuda.empty_cache()
        
        print(f"✅ [Voice Node] Alert successfully generated at {output_file}")
        return output_file
        
    except Exception as e:
        # Emergency VRAM cleanup in case of failure (e.g., OOM error during synthesis)
        torch.cuda.empty_cache()
        error_msg = f"LOCAL TTS ERROR: Failed to synthesize speech. Details: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

# =====================================================================
# LOCAL TESTING BLOCK (Comment out in production)
# =====================================================================
# if __name__ == "__main__":
#     test_note = "Cảnh báo y tế: Phát hiện rủi ro tương tác thuốc."
#     res = generate_clinical_voice_alert.invoke(test_note)
#     print(res)
