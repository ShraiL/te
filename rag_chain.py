# ═══════════════════════════════════════════════════════════════
# RAG_CHAIN.PY - Ask questions about your documents
# RUN THIS AFTER ingest.py!
# ═══════════════════════════════════════════════════════════════

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CHANGE THESE IF NEEDED                                       ║
# ╚═══════════════════════════════════════════════════════════════╝
DB_DIR = 'vectorstore'               # Must match ingest.py
EMBEDDING_MODEL = 'nomic-embed-text' # Must match ingest.py
LLM_MODEL = 'llama3.1'               # Model for answering
NUM_RESULTS = 3                      # How many chunks to retrieve


# Load vectorstore
emb = OllamaEmbeddings(model=EMBEDDING_MODEL)
db = Chroma(persist_directory=DB_DIR, embedding_function=emb)

# Create retriever
retriever = db.as_retriever(search_kwargs={"k": NUM_RESULTS})

# Load LLM
llm = Ollama(model=LLM_MODEL)

# Prompt template
prompt = PromptTemplate(
    template="""Use the following context to answer the question.
If you don't know the answer, say "I don't know."

Context: {context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)


def ask(question):
    """Ask a question and get an answer based on documents"""
    # 1. Retrieve relevant chunks
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. Format prompt with context + question
    formatted_prompt = prompt.format(context=context, question=question)
    
    # 3. Get answer from LLM
    answer = llm.invoke(formatted_prompt)
    
    return answer


if __name__ == '__main__':
    # ╔═══════════════════════════════════════════════════════════╗
    # ║  CHANGE YOUR QUESTION HERE                                ║
    # ╚═══════════════════════════════════════════════════════════╝
    question = "What is machine learning?"
    
    print(f"Question: {question}")
    print("-" * 50)
    answer = ask(question)
    print(f"Answer: {answer}")