import os
import torch
import logging
from pathlib import Path
from langchain.tools import tool
import soundfile as sf
from voxcpm import VoxCPM
from typing import Optional

logger = logging.getLogger(__name__)

# =====================================================================
# MODEL DOWNLOAD & SETUP CONFIGURATION
# =====================================================================
HF_REPO_ID = "JayLL13/VoxCPM-1.5-VN"

# Dynamic Pathing: Calculate root directory safely regardless of execution folder
BASE_DIR = Path(__file__).resolve().parent.parent.parent
VOICE_OUT_DIR = BASE_DIR / "data" / "voice_alerts"


@tool
def generate_clinical_voice_alert(
    clinical_note: str,
    prompt_wav_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
) -> str:
    """Use this tool to synthesize a voice alert from the clinical reasoning text."""
    current_model = None

    try:
        logger.info("🎙️ [Voice Node] Initiating local TTS synthesis...")
        logger.info("🧹 [Memory Manager] Clearing VRAM to prepare for VoxCPM...")
        torch.cuda.empty_cache()

        logger.info(
            "📥 [Voice Node] Loading VoxCPM model into GPU. This may take a moment..."
        )
        current_model = VoxCPM.from_pretrained(HF_REPO_ID)

        VOICE_OUT_DIR.mkdir(parents=True, exist_ok=True)
        output_file = str(VOICE_OUT_DIR / "clinical_alert.wav")

        logger.info(
            f"🔊 [Voice Node] Synthesizing audio for: '{clinical_note[:50]}...'"
        )

        wav = current_model.generate(
            text=clinical_note,
            prompt_wav_path=prompt_wav_path if prompt_wav_path else None,
            prompt_text=prompt_text if prompt_text else None,
            cfg_value=2.0,
            inference_timesteps=10,
            normalize=False,
            denoise=False,
            retry_badcase=True,
            retry_badcase_max_times=3,
            retry_badcase_ratio_threshold=6.0,
        )

        sf.write(output_file, wav, current_model.tts_model.sample_rate)
        logger.info(f"✅ [Voice Node] Alert successfully generated at {output_file}")

        return output_file

    except Exception as e:
        error_msg = f"LOCAL TTS ERROR: Failed to synthesize speech. Details: {str(e)}"
        logger.error(f"❌ {error_msg}", exc_info=True)
        return error_msg

    finally:
        # ==========================================================
        # CRITICAL STEP: Purge the model from VRAM instantly
        # Guaranteed to run even if model.generate() crashes!
        # ==========================================================
        logger.info(
            "🧹 [Memory Manager] Unloading VoxCPM and freeing up VRAM in finally block..."
        )
        if current_model is not None:
            del current_model
        if "wav" in locals():
            del wav
        torch.cuda.empty_cache()
