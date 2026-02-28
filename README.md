# 🏥 OmniMed-Agent-OS

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange)
![Llama-3](https://img.shields.io/badge/Llama_3-8B_4--bit-green)
![VRAM](https://img.shields.io/badge/Optimized_for-16GB_VRAM-red)

**OmniMed-Agent-OS** is a robust, multimodal, and fully localized (Vietnamese) AI Medical Assistant designed to run entirely completely offline on highly constrained hardware (e.g., a single 16GB VRAM GPU like the Nvidia T4).

## 🌟 Key Features & Engineering Highlights

- **🧠 Dynamic VRAM Management:** Implements strict dynamic model loading and purging. Loads Vision, RAG, Reasoning (Llama-3 8B Unsloth 4-bit), and Voice TTS (VoxCPM) models sequentially into the GPU, ensuring the system never exceeds 16GB VRAM.
- **👁️ Medical Document OCR (Docling):** Accurately extracts complex layouts and medical tables (e.g., blood test results) into structured Markdown format, avoiding LLM hallucinations.
- **📚 Localized Medical RAG:** Uses the `bkai-foundation-models/vietnamese-bi-encoder` to perform highly accurate semantic searches over Vietnamese Electronic Health Records (EHR) and clinical guidelines using ChromaDB.
- **🗣️ On-Premise Voice Synthesis:** Synthesizes clinical alerts using a local VoxCPM instance without relying on external cloud APIs, ensuring HIPAA/GDPR compliance for patient data.
- **⚙️ Deterministic State Machine:** Orchestrated using **LangGraph** to ensure a strict, predictable execution pipeline: `Vision -> Retrieval -> Reasoning -> Voice Alert`.

## 🏗️ System Architecture

1. **Vision Node:** Parses attached scanned PDFs/Images.
2. **RAG Node:** Fetches relevant patient history based on the doctor's query.
3. **Reasoning Node:** LLM synthesizes OCR + RAG data to provide clinical insights.
4. **Voice Node:** Converts the critical parts of the diagnosis into an audio alert.

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone [https://github.com/cantricao/OmniMed-Agent-OS.git](https://github.com/cantricao/OmniMed-Agent-OS.git)
cd OmniMed-Agent-OS
pip install -r requirements.txt
```

**2. Run the Workflow** Ensure your .env contains your GOOGLE\_API\_KEY or GEMINI\_API\_KEY.

```bash
# Execute the LangGraph pipeline
python src/main_workflow.py
```

## 🛡️ Security & Privacy
All models and vector databases are hosted locally. Zero patient data is transmitted to external endpoints like OpenAI or Google Cloud, maintaining strict medical data confidentiality.

👨‍💻 About the Author
----------------------

**Tri Cao Can** AI Engineer & Data Analyst | Biomedical Data Science Specialist

With over 3 years of professional experience in developing machine learning models and automated data pipelines , I hold a Master of Data Analytics from QUT. My core focus lies at the intersection of computational biology, genomic data analysis, and scalable AI infrastructure. This repository reflects my passion for building secure, data-driven systems that solve complex translational challenges.

* **Email:** cantricao@gmail.com
* **LinkedIn:** [linkedin.com/in/cao-tri-can](https://www.linkedin.com/in/cao-tri-can-08188b21b/)
* **Portfolio:** [Notion Portfolio](https://cumbersome-tachometer-03f.notion.site/)
* **GitHub:** [github.com/cantricao](http://github.com/cantricao)
