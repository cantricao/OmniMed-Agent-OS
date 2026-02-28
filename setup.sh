#!/bin/bash

# =====================================================================
# OMNIMED-AGENT-OS: ENVIRONMENT SETUP SCRIPT
# This script prepares the Linux environment and installs all dependencies.
# =====================================================================

echo "🚀 [Setup] Initializing environment setup for OmniMed..."

# 1. Install system-level dependencies (FFmpeg for audio processing)
echo "📦 [Setup] Updating package lists and installing FFmpeg (System Audio Codec)..."
sudo apt-get update -qq
sudo apt-get install -y ffmpeg

# 2. Install Python dependencies
# Assuming you have a requirements.txt, if not, it installs gradio directly
echo "🐍 [Setup] Installing Python dependencies..."
pip install -r requirements.txt -q

echo "✅ [Setup] Environment is fully prepared. You can now launch the application!"
