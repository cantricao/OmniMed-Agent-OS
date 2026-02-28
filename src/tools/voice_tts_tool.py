import os
import torch
import torchaudio
from huggingface_hub import snapshot_download
from langchain.tools import tool

# =====================================================================
# MODULE IMPORT
# =====================================================================
# Attempt to import the core Voice Cloning module from your local directory. 
# Adjust 'VoxCPMModel' based on the exact class name in your project.
try:
    from voxcpm import VoxCPMModel 
except ImportError:
    print("⚠️ WARNING: VoxCPM core libraries are not installed in this environment.")

# =====================================================================
# MODEL DOWNLOAD & SETUP CONFIGURATION
# =====================================================================
# Define the HuggingFace repository ID for VoxCPM
# Change this if you are using a specific or fine-tuned variant
HF_REPO_ID = "Boya-S/VoxCPM"
LOCAL_MODEL_DIR = "data/models/voxcpm"

def ensure_model_downloaded() -> str:
    """
    Downloads the VoxCPM model checkpoints from Hugging Face if they are not 
    already present locally. This prevents re-downloading large files on every run.
    """
    # Check if directory exists and is not empty
    if not os.path.exists(LOCAL_MODEL_DIR) or not os.listdir(LOCAL_MODEL_DIR):
        print(f"📥 [Voice Setup] Downloading VoxCPM weights from Hugging Face ({HF_REPO_ID})...")
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        # Download the model weights and configuration files
        snapshot_download(repo_id=HF_REPO_ID, local_dir=LOCAL_MODEL_DIR)
        print("✅ [Voice Setup] Model checkpoints downloaded successfully.")
    else:
        print("✅ [Voice Setup] Local model checkpoint found. Skipping download.")
        
    return LOCAL_MODEL_DIR

# =====================================================================
# DIRECT TTS TOOL (VoxCPM Local Inference with Dynamic VRAM Management)
# =====================================================================

@tool
def generate_clinical_voice_alert(clinical_note: str) -> str:
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
        checkpoint_path = ensure_model_downloaded()
        
        print("🧹 [Memory Manager] Clearing VRAM to prepare for VoxCPM...")
        # 1. Forcefully release unreferenced GPU memory from previous nodes (Vision/LLM)
        torch.cuda.empty_cache() 
        
        # 2. Dynamically load the VoxCPM model into GPU ONLY when called
        print("📥 [Voice Node] Loading VoxCPM model into GPU. This may take a moment...")
        
        # Instantiate the model from the local downloaded path
        current_model = VoxCPMModel.from_pretrained(checkpoint_path).to("cuda")
        
        # Ensure the output directory exists
        output_dir = "data/voice_alerts"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "clinical_alert.wav")
        
        # 3. Execute the inference to generate speech
        print(f"🔊 [Voice Node] Synthesizing audio for: '{clinical_note[:50]}...'")
        
        wav_tensor = current_model.generate(
            text=clinical_note,
            cfg_value=2.0,                  # Control scale for generation quality
            inference_timesteps=25,         # Number of steps for inference
            normalize=False,                # External text normalization
            denoise=False                   # Post-processing denoise filter
        )
        
        # Retrieve the sample rate dynamically from the model's inner configuration
        sample_rate = current_model.tts_model.sample_rate if hasattr(current_model, 'tts_model') else 16000
        
        # Ensure the tensor is 2D [channels, time] as required by torchaudio
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
            
        # Save the synthesized waveform to a physical audio file on the local disk
        torchaudio.save(output_file, wav_tensor.cpu(), sample_rate=sample_rate)
        
        # ==========================================================
        # 4. CRITICAL STEP: Purge the model from VRAM instantly
        # ==========================================================
        print("🧹 [Memory Manager] Unloading VoxCPM and freeing up VRAM...")
        del current_model
        del wav_tensor
        torch.cuda.empty_cache()
        
        print(f"✅ [Voice Node] Alert successfully generated at {output_file}")
        return f"SUCCESS: Voice alert generated. Audio file saved at: {output_file}"
        
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
