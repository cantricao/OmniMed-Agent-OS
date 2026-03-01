# =====================================================================
# OMNIMED-AGENT-OS: ENTERPRISE DOCKERFILE (GPU ENABLED)
# =====================================================================
# 1. Base Image: We use the official PyTorch image with CUDA 12.1 pre-installed
# This saves us from compiling heavy GPU drivers from scratch.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# 2. Environment Variables
# Prevent interactive prompts from blocking the build process
ENV DEBIAN_FRONTEND=noninteractive
# Force Python to print logs immediately (useful for real-time tracking)
ENV PYTHONUNBUFFERED=1
# Allow Gradio to be accessed from outside the container
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# 3. Install System-Level Dependencies
# We clean up apt cache (rm -rf) to keep the Docker image size small
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the Working Directory inside the container
WORKDIR /app

# 5. Dependency Caching Strategy
# Copy ONLY the dependency files first. Docker caches this layer, 
# so rebuilding the image is lightning fast if you only change your code (not your pip packages).
COPY requirements.txt setup.sh ./
COPY scripts/ ./scripts/

# 6. Install Python Dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Trigger dynamic Unsloth installation script
    python scripts/install_unsloth.py

# 7. Copy the Application Source Code
COPY src/ ./src/
COPY data/ ./data/
COPY app.py .env.example ./

# 8. Expose the Gradio Web UI Port
EXPOSE 7860

# 9. Define the default execution command
CMD ["python", "app.py"]
