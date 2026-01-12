# ML Final Exam - Ready Templates 🎯

## Quick Setup (Before Exam)

### 1. Install Packages
```bash
pip install langchain langchain-community langchain-text-splitters langchain-ollama chromadb transformers pillow torch
```

### 2. Download Ollama Models
```bash
ollama pull nomic-embed-text
ollama pull llama3.1
```

### 3. Make Sure Ollama is Running
```bash
ollama serve
```

---

## TASK 1: RAG Pipeline (Ollama + LangChain)

### Step 1: Put your text files in `data/` folder

### Step 2: Run ingest.py
```bash
python ingest.py
```
Output: "Loaded X documents" → "Created Y chunks" → "DONE!"

### Step 3: Run rag_chain.py
```bash
python rag_chain.py
```
Change the question in the file if needed.

---

## TASK 2: BLIP (Image to Text)

### Step 1: Put your image in `images/` folder

### Step 2: Change IMAGE_PATH in blip_caption.py

### Step 3: Run
```bash
python blip_caption.py
```

---

## What to Change During Exam

| File | What to Change |
|------|----------------|
| `ingest.py` | `DATA_DIR`, `CHUNK_SIZE`, `EMBEDDING_MODEL` if asked |
| `rag_chain.py` | `question` variable |
| `blip_caption.py` | `IMAGE_PATH`, `TEXT_PROMPT`, `QUESTION` |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| "Ollama connection refused" | Run `ollama serve` in terminal |
| "Model not found" | Run `ollama pull model_name` |
| "File not found" | Check your paths |
| Circular import | Don't name files `ollama.py` or `transformers.py` |
