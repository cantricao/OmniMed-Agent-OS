import os
import re
import subprocess
import sys

# =====================================================================
# UNSLOTH DYNAMIC INSTALLER
# Resolves strict version dependencies between PyTorch and Xformers
# =====================================================================

def run_cmd(cmd: str):
    """Executes shell commands safely and prints the output."""
    print(f"⚙️ [Unsloth Setup] Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def install_unsloth():
    print("🦥 [Unsloth Setup] Initializing dynamic installation for Unsloth...")
    
    # Check if the environment is Google Colab
    env_keys = "".join(os.environ.keys())
    is_colab = "COLAB_" in env_keys
    
    if not is_colab:
        print("💻 [Unsloth Setup] Detected Local/Standard Cloud environment.")
        run_cmd("pip install unsloth -q")
    else:
        print("☁️ [Unsloth Setup] Detected Google Colab environment. Matching Torch versions...")
        # Ensure torch is installed before checking version
        try:
            import torch
        except ImportError:
            run_cmd("pip install torch -q")
            import torch
            
        # Extract base Torch version (e.g., '2.10' from '2.10.0+cu128')
        v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
        
        # Map Torch version to exact compatible Xformers wheel
        xformers_map = {'2.10': '0.0.34', '2.9': '0.0.33.post1', '2.8': '0.0.32.post2'}
        xformers_ver = xformers_map.get(v, "0.0.34")
        xformers_pkg = f"xformers=={xformers_ver}"
        
        print(f"🧩 [Unsloth Setup] Selected {xformers_pkg} for Torch {v}")
        
        # Execute complex dependency installations
        run_cmd('pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer -q')
        run_cmd(f'pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers_pkg} peft trl triton unsloth -q')
    
    # Force specific versions of transformers and trl to ensure compatibility
    print("📦 [Unsloth Setup] Forcing specific versions for transformers and trl...")
    run_cmd("pip install transformers==4.56.2 -q")
    run_cmd("pip install --no-deps trl==0.22.2 -q")
    
    print("✅ [Unsloth Setup] Unsloth and dependencies installed successfully!")

if __name__ == "__main__":
    install_unsloth()
