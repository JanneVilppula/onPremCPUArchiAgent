import json
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

facts = []
try:
    with open('facts.json', 'r', encoding='utf-8') as f:
        facts = json.load(f)
    print(f"Loaded {len(facts)} facts from facts.json")
except FileNotFoundError:
    print(f"Error: facts.json not found. Run archiExportIngestor.py to generate facts.json")
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")

lc_documents = []
for fact in facts:
    metadata = {k: v for k, v in fact.items() if k not in ['id', 'text']}
    lc_documents.append(
        Document(
            page_content=fact['text'],
            metadata=metadata
        )
    )

try:
    print(f"\nInitialising Ollama Embeddings with nomic-embed-text")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    _ = embeddings.embed_query("test Ollama server on")
    print("nomic-embed-text initialised")
except Exception as e:
    print(f"Error initialising nomic-embed-text {e}")

try:
    vectorstore = FAISS.from_documents(lc_documents,embeddings)
    vectorstore.save_local("facts_faiss_index")
    print("FAISS index created and saved")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    vectorstore = None