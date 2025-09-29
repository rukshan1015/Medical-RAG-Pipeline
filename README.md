
# Medical RAG — Simple RAG Demo (with data + notes + vectors)

A minimal Retrieval-Augmented Generation (RAG) pipeline for clinical-style questions over synthetic EHR-like data.  
This repo includes **original CSV data**, **generated patient notes**, and a **prebuilt Chroma vector DB** so you can test the app right away—no preprocessing required.

---

## What’s inside
- **`Medical_RAG_pipeline.py`** — Gradio chat app (ask questions; answers grounded with retrieved snippets)
- **`patient_notes_gen.py`** — builds per-patient notes from CSV tables *(for reference / reproducibility)*
- **`patient_vector_embd.py`** — splits notes, embeds with `sentence-transformers/paraphrase-mpnet-base-v2`, persists to **Chroma** *(for reference / reproducibility)*
- **Assets** — original CSVs, generated notes, and a ready-to-use **Chroma** vector store folder

**Tech:** Hugging Face embeddings (`paraphrase-mpnet-base-v2`), ChromaDB, LangChain (ConversationalRetrievalChain), OpenAI-compatible chat model (e.g., `gpt-5-mini`), Gradio UI.

---

## Download the prebuilt vector DB (Google Drive) — gdown

Use `gdown` to fetch the **enc_records** folder locally, then point the app to it.

```bash
pip install gdown

# Download the shared folder into ./enc_records
gdown --folder "https://drive.google.com/drive/folders/12Wyn_LDFsRJrRz2vHV4wnP8bwnBOMMho?usp=sharing" -O enc_records
```

## Quick start (zero setup)

> **Fast path for recruiters/testers:**  
> **Download the vector embeddings folder from Google drive (as shown above)** and **`Medical_RAG_pipeline.py`**, install requirements, and run the app.  
> The other scripts are **for information only**—you don’t need them to test.

1) **Install**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
