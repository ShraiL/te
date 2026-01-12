ML Final Exam - Ready Templates 🎯
Quick Setup (Before Exam)
1. Install Packages
bashpip install langchain langchain-community langchain-text-splitters langchain-ollama chromadb transformers pillow torch
2. Download Ollama Models
bashollama pull nomic-embed-text
ollama pull llama3.1
3. Make Sure Ollama is Running
bashollama serve

TASK 1: RAG Pipeline (Ollama + LangChain)
Step 1: Put your text files in data/ folder
Step 2: Run ingest.py
bashpython ingest.py
Output: "Loaded X documents" → "Created Y chunks" → "DONE!"
Step 3: Run rag_chain.py
bashpython rag_chain.py
Change the question in the file if needed.

TASK 2: BLIP (Image to Text)
Step 1: Put your image in images/ folder
Step 2: Change IMAGE_PATH in blip_caption.py
Step 3: Run
bashpython blip_caption.py

What to Change During Exam
FileWhat to Changeingest.pyDATA_DIR, CHUNK_SIZE, EMBEDDING_MODEL if askedrag_chain.pyquestion variableblip_caption.pyIMAGE_PATH, TEXT_PROMPT, QUESTION

Troubleshooting
ErrorFix"Ollama connection refused"Run ollama serve in terminal"Model not found"Run ollama pull model_name"File not found"Check your pathsCircular importDon't name files ollama.py or transformers.py
