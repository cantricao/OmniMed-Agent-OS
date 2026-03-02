# 🏥 OmniMed-Agent-OS
![CI/CD Pipeline](https://github.com/cantricao/OmniMed-Agent-OS/actions/workflows/ci.yml/badge.svg)
<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.10+-EE4C2C.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Framework-LangGraph-000000.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/VectorDB-Chroma-4B0082.svg" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Optimization-Unsloth-FF9900.svg" alt="Unsloth">
  <img src="https://img.shields.io/badge/Deployment-Docker-2496ED.svg?logo=docker&logoColor=white" alt="Docker">
</div>

*An Edge-Deployed, Privacy-First Multimodal Medical AI Assistant with RAG and Zero-Shot Voice Cloning.*

## 📌 Executive Summary
**OmniMed-Agent-OS** is an advanced, fully localized AI agent designed to process unstandardized medical documents (receipts, prescriptions), reason over a custom medical corpus, and synthesize natural voice alerts. 

Built with strict privacy constraints and biomedical data analytics principles, the system runs entirely on local hardware (optimized for constrained GPUs like the Tesla T4) without transmitting sensitive Protected Health Information (PHI) to external APIs.

## 🏗️ System Architecture

OmniMed-Agent-OS utilizes a **State-Driven Agentic Workflow** powered by LangGraph. The system ensures medical data integrity through a strict pipeline with a built-in privacy layer.

```mermaid
graph TD
    %% Input Layer
    User((Doctor/User)) -->|Upload Image/PDF| App[Gradio Web UI]
    User -->|Voice/Text Query| App

    subgraph "OmniMed Core Engine (LangGraph)"
        %% Process Nodes
        A[Vision_OCR Node] -->|Raw Text| B(Data_Sanitization Node)
        B -->|Redacted Text| C(EHR_RAG Node)
        
        %% Database & Models
        D[(ChromaDB Vector Store)] <-->|Context Retrieval| C
        C --> E(Clinical_Reasoning Node)
        
        %% Security & Control
        E -->|Final Diagnosis| F{Human-in-the-Loop}
        
        %% Outputs
        F -->|Approved| G[Voice_Alert Node]
        F -->|Rejected| H[System Reset/Retry]
    end

    %% Final Outputs
    G -->|Audio Stream| Out1((Voice Notification))
    E -->|Text Stream| Out2((Clinical Report))

    %% Styling
    style B fill:#f96,stroke:#333,stroke-width:2px
    style F fill:#32CD32,stroke:#333,stroke-width:2px
    style D fill:#6495ED,stroke:#333,stroke-width:2px
  ```

## 🎯 Expected Output & Demo

<div align="center">
  <h3>📝 Input: Sample Medical Receipt</h3>
  <img src="data/images/test_receipt.jpg" alt="Test receipt" width="400">
  
  <br><br> <h3>▶️ Output: OmniMed-Agent-OS Execution & Voice Alert</h3>
  <video src="https://github.com/user-attachments/assets/232bcfed-4209-462b-bb0c-5246221a543e" controls="controls" width="800"></video>
</div>

**Sample Clinical Reasoning Result:**
```text
==================================================
📋 OMNIMED FINAL CLINICAL REPORT (UI)
==================================================
Danh sách các mặt hàng/dịch vụ và đơn giá tương ứng:

* Sultamicillin375mgUNASYN: 08 viên, giá không có thông tin
* NEXTGCAL: 30 viên, giá không có thông tin
* HEMOQMOM: 30 viên, giá không có thông tin
* Povidine10%90ML: 01 chai, giá không có thông tin
* Bocham soc ron: 01 bo, giá không có thông tin

Tổng số tiền phải thanh toán: Không có thông tin

==================================================
🔊 OMNIMED VOICE SUMMARY (TTS)
==================================================
Phân tích hoàn tất. Có năm loại thuốc. Bác sĩ vui lòng xem chi tiết trên màn hình.
```

## 🌟 Core Architecture (Agentic Workflow)
The system is orchestrated using **LangGraph**, smoothly transitioning a unified `MedicalState` through four specialized nodes:

1. **👁️ Vision Node (RapidOCR):** Extracts structured text, tabular data, and complex layouts from noisy medical images.
2. **🔍 RAG Node (ChromaDB + HF Bi-encoder):** Performs semantic search against a locally embedded vector database of comprehensive medical QA records (ViHealthQA). Built with robust batch-ingestion scripts to handle large-scale data pipelines without memory overflow.
3. **🧠 Clinical Reasoning Node (Unsloth + Llama-3 8B):** Loads quantized models dynamically to analyze symptoms and OCR data. Advanced prompt engineering enforces strict anti-hallucination rules, prevents data fabrication, and automatically restores missing language diacritics.
4. **🎙️ Voice Alert Node (VoxCPM):** Synthesizes a concise clinical summary for healthcare professionals. Features **Zero-Shot Voice Cloning** to mimic specific speaker profiles, utilizing dynamic hardware dispatch to maximize GPU efficiency.

## 🛠️ Project Structure
```text
OmniMed-Agent-OS/
├── app.py                      # Main Gradio Web UI with Model A/B Testing & Voice Cloning
├── setup.sh                    # Automated bash script for System/OS dependency resolution
├── requirements.txt            # Python dependencies
├── src/
│   ├── main_workflow.py        # LangGraph State & Node Orchestration
│   ├── core/
│   │   ├── ingest_real_data.py # Robust ETL pipeline for batch-vectorizing datasets
│   │   └── local_llm.py        # Unsloth LLM initialization and inference logic
│   └── tools/
│       ├── ocr_vision_tool.py  # RapidOCR wrappers
│       └── voice_tts_tool.py   # VoxCPM Audio backend and dynamic tensor dispatch
└── tests/
    └── test_rag_db.py          # Validation scripts for ChromaDB semantic search accuracy
```
## 🚀 Getting Started
**1. Automated Environment Setup:**
Say goodbye to dependency hell. The provided setup script automatically installs OS-level codecs (`ffmpeg`, `libsndfile1`) and perfectly aligns PyTorch versions with hardware-accelerated libraries.
```bash
chmod +x setup.sh
./setup.sh
```

**2. Data Ingestion (Vector Database):**
Build your local knowledge base. This script automatically downloads the `tarudesu/ViHealthQA` dataset from HuggingFace and safely ingests records into ChromaDB using memory-safe batching. (Note: The raw CSV is ignored in version control to maintain repository efficiency).
```bash
python src/core/ingest_real_data.py
```

**3. Quality Assurance / Unit Testing:**
Verify that your semantic search engine is populated and functioning correctly before launching the main application.
```bash
python python tests/test_rag_db.py
```

**4. Launch the AI OS**
Fire up the full pipeline. The Gradio interface allows you to upload documents, select different open-source reasoning models dynamically, and even test voice cloning with reference audio.
```bash
python app.py
```

**5. Advanced Usage (CLI / Headless Mode)**
For developers or deployment on GUI-less Linux servers, you can bypass the Gradio interface and execute the LangGraph workflow directly via the command line interface (CLI). This is ideal for CI/CD pipeline testing or batch processing.

```bash
python -m src.main_workflow
```
---

## 🐳 One-Click Local Deployment (Docker)

To provide a seamless, environment-agnostic experience, OmniMed-Agent-OS is fully containerized. You do not need to deal with Python environments or dependency conflicts.

### Prerequisites
* [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (strictly required for GPU acceleration)

### Quick Start Guide

**1. Clone the repository:**
```bash
git clone [https://github.com/cantricao/OmniMed-Agent-OS.git](https://github.com/cantricao/OmniMed-Agent-OS.git)
cd OmniMed-Agent-OS
```

**2. Launch the AI Engine:**
Deploy the entire application (OCR, LangGraph, LLM, TTS) with a single command:
```bash
docker-compose up -d --build
```

**3. Access the Interface:**
Verify that your semantic search engine is populated and functioning correctly before launching the main application.
👉 **http://localhost:7860**
*Note: The first run may take a few minutes to download the base image and AI model weights. Subsequent runs will be instantaneous.*

### 🔧 Configuration Editing
Need to change the AI's behavior or language? You don't need to rebuild the Docker image! Simply edit the configs/system_config.yaml file on your host machine, and the changes will reflect inside the container automatically.

---

## 🔬 Technical Highlights for Data/ML Engineers
* **Memory Management:** Implemented strict VRAM clearing (`torch.cuda.empty_cache()`) between pipeline steps, allowing heavy OCR, RAG, 8B LLMs, and TTS models to run sequentially on a single 16GB VRAM GPU.
* **Anti-Hallucination Guardrails:** Strict prompt engineering ensures the AI only extracts explicitly stated medical pricing and quantities, translating all operational metadata strictly to the target language without arbitrary conversational filler.
* **Scalable ETL:** RAG ingestion is decoupled from the main app, utilizing `tqdm` tracking and batch-chunking, preparing the system for enterprise-scale electronic health records (EHR) databases.
* **A/B Testing Ready:** The UI architecture exposes model selection states natively, allowing developers to hot-swap reasoning models (e.g., Llama-3, Mistral, Gemma) instantly via the Gradio interface without altering core logic.
---

## 🗺️ Roadmap & Future Enhancements
While the current OS operates efficiently on edge devices, the architecture is designed to scale:
* **[] FHIR/HL7 Integration:** Standardize OCR and reasoning outputs to comply with international electronic health record interoperability standards.
* **[] Real-time Streaming TTS:** Implement chunk-based audio streaming to reduce Time-To-First-Audio (TTFA) for voice alerts.
* **[] DICOM Image Support:** Expand the Vision Node to support standard medical imaging formats natively alongside standard OCR.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page. If you are looking to run this in a production environment, please ensure you have at least a 16GB VRAM GPU (e.g., NVIDIA T4, RTX 4080, or A10G) to handle the sequential multimodal execution.

---

# 📜 License
Distributed under the MIT License. See LICENSE for more information.

---


👨‍💻 About the Author
----------------------

**Tri Cao Can** AI Engineer & Data Analyst | Biomedical Data Science Specialist

With a Master of Data Analytics specializing in Biomedical Data Analytics from the Queensland University of Technology (QUT), I bridge the gap between complex machine learning architectures and practical, privacy-first healthcare solutions. Based in Australia, I specialize in building end-to-end data pipelines, optimizing local LLM deployments, and developing robust AI systems for real-world clinical and enterprise environments.

* **Open to opportunities:** Actively seeking Data Scientist, ML Engineer, or Data Engineer roles in Australia, as well as high-impact freelance projects.

* **Let's connect:** Feel free to reach out via GitHub or LinkedIn to discuss AI in healthcare, hardware optimization, or enterprise data pipelines.

* **Email:** cantricao@gmail.com
* **LinkedIn:** [linkedin.com/in/cao-tri-can](https://www.linkedin.com/in/cao-tri-can-08188b21b/)
* **Portfolio:** [Notion Portfolio](https://cumbersome-tachometer-03f.notion.site/)
* **GitHub:** [github.com/cantricao](http://github.com/cantricao)
