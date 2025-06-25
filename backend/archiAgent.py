from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "deepseek-r1"
FAISS_INDEX_PATH = "facts_faiss_index"

try:
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
    _ = embeddings.embed_query("test")
    print(f"{OLLAMA_EMBEDDING_MODEL} initialised")
except Exception as e:
    print(f"Error, FAISS index not found")

try:
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH,embeddings,allow_dangerous_deserialization=True)
    print(f"FAISS index loaded")
except Exception as e:
    print(f"Error loading FAISS index: '{e}'")

try:
    llm = OllamaLLM(model=OLLAMA_LLM_MODEL)
    print(f"{OLLAMA_LLM_MODEL} initialised")
except Exception as e:
    print(f"Error intitalising {OLLAMA_LLM_MODEL}: '{e}'")

retriver = vectorstore.as_retriever(search_kwargs={"k": 5})
prompt = ChatPromptTemplate.from_template("""
You are an expert enterprise architect chatbot providing information based on the provided ArchiMate model context.
Answer the user's question using ONLY the provided architectural context.
If the context does not provide enough information to answer the question, state that clearly.
Avoid making up information.
Keep your answer concise and focused on the architectural details.

Context:
{context}

Question:
{question}
""")

rag_chain = (
    {"context": retriver, "question": RunnablePassthrough()}
     | prompt
     | llm
     | StrOutputParser()
)

print("\n--- ArchiMate Chatbot Ready ---")
print("Type your questions about the architecture, or 'exit' to quit.")

while True:
    user_query = input("\nYour question: ")
    if user_query.lower() == 'exit':
        print("Exiting chatbot. Goodbye!")
        break

    print("Searching and generating response...")
    try:
        response = rag_chain.invoke(user_query)
        print("\n--- Answer ---")
        print(response)
    except Exception as e:
        print(f"An error occurred during response generation: {e}")
        print("Please check your Ollama server status and model configuration.")
