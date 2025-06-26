from typing import List, Dict, Optional
from langchain_core.language_models.base import BaseLanguageModel, Runnable

import sys
import time
import threading
from pathlib import Path

import xml.etree.ElementTree as et
import pandas as pd
import json

from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from operator import itemgetter 

OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "mistral:7b"
FAISS_INDEX_PATH = "facts_faiss_index"
try:
    ARCHI_XML_PATH = max(list(Path(__file__).parent.glob('*.xml')), key=lambda p: p.stat().st_mtime)
    tree = et.parse(ARCHI_XML_PATH)
    root = tree.getroot()
    et_elements = root[2]
    et_relationships = root[3]
    ARCHIMATE_NAMESPACE = "{http://www.opengroup.org/xsd/archimate/3.0/}"
    XSI_NAMESPACE = "{http://www.w3.org/2001/XMLSchema-instance}"
except:
    print(f"Cannot find Archi-export XML. Ensure the python-script is in the same folder with the sole XML-file in that folder.")

class Spinner:
    def __init__(self, message="Processing...", delay=0.1):
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.delay = delay
        self.message = message
        self.running = False
        self.spinner_thread = None
        self.stop_event = threading.Event()

    def _spinning_thread(self):
        thread_start_time = time.time()
        i = 0
        while not self.stop_event.is_set():
            char = self.spinner_chars[i % len(self.spinner_chars)]
            elapsed_seconds = int(time.time() - thread_start_time)
            minutes = elapsed_seconds // 60
            seconds = elapsed_seconds % 60

            time_str = f"{minutes:02d}:{seconds:02d}"
            full_line = f"\r{char} {self.message} ({time_str})"

            sys.stdout.write(full_line)
            sys.stdout.flush()

            sys.stdout.write(' ' * (max(0, self.last_line_len - len(full_line) + 1)) + "\r")
            self.last_line_len = len(full_line)

            time.sleep(self.delay)
            i += 1

    def start(self):
        if not self.running:
            self.running = True
            self.stop_event.clear()
            self.start_time = time.time()
            self.last_line_len = 0
            self.spinner_thread = threading.Thread(target=self._spinning_thread)
            self.spinner_thread.daemon = True
            self.spinner_thread.start()

    def stop(self):
        if self.running:
            self.stop_event.set()
            if self.spinner_thread and self.spinner_thread.is_alive():
                self.spinner_thread.join(timeout=0.5)
            sys.stdout.write("\r" + " " * (len(self.message) + 5) + "\r")
            sys.stdout.flush()
            self.running = False

def should_vector_db_update() -> bool:
    user_input = input("Do you want to update the vector database before starting? Answer Y or N: ")
    if user_input.lower() in ('y', 'yes'):
        return True
    else:
        return False 

def df_from_XML(xml) -> pd.DataFrame:
    list = []
    for entry in xml:
        cleaned_entry = {}
        for key, value in entry.items():
            key = key[key.rfind("}")+1:]
            cleaned_entry[key] = value
        for child in entry:
            if child.tag.count("name") > 0:
                cleaned_entry["name"] = child.text
            else:
                cleaned_entry["documentation"] = child.text
        list.append(cleaned_entry)
    df = pd.DataFrame(list)
    return df

def facts_from_df(elements, realtionships) -> List[str]:
    outgoing = pd.merge(
        realtionships, 
        elements.rename(columns={
            'identifier': 'source_id',
            'type': 'source_type',
            'name': 'source_name',
            'documentation': 'source_documentation'
        }), 
        left_on="source", 
        right_on="source_id", 
        how="left")
    union = pd.merge(        
        outgoing, 
        elements.rename(columns={
            'identifier': 'target_id',
            'type': 'target_type',
            'name': 'target_name',
            'documentation': 'target_documentation'
        }), 
        left_on="target", 
        right_on="target_id", 
        how="left")
    facts = []
    for row in union.itertuples():
        facts.append(f"{row.source_name} ({row.source_type}) is the source of a {row.type} to {row.target_name} ({row.target_type})")
        facts.append(f"{row.target_name} ({row.target_type}) is the target of a {row.type} to {row.source_name} ({row.source_type})")
    for row in elements.itertuples():
        facts.append(f"{row.name} ({row.type}) is described as: '{row.documentation}'")
    return facts

def combine_facts_to_langchain_document(facts) -> List[Document]:
    result = []
    for fact in facts:
        doc = Document(page_content=fact)
        result.append(doc)
    return result

def vectorise_langchain_document_into_faiss(document, OLLAMA_EMBEDDING_MODEL, FAISS_INDEX_PATH) -> FAISS:
    print("Starting to initialise FAISS. This may take a minute")
    spinner = Spinner("Intialising FAISS...", delay=0.1)
    spinner.start()
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(document, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    loaded_vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    spinner.stop()
    print("FAISS index loaded")
    return loaded_vectorstore

def load_faiss_index(OLLAMA_EMBEDDING_MODEL, FAISS_INDEX_PATH) -> Optional[FAISS]:
    print("Starting to initialise FAISS.")
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
    _ = embeddings.embed_query("test")
    loaded_vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("FAISS index loaded")
    return loaded_vectorstore

def ask_llm_to_use(DEFAULT_LLM_MODEL) -> str:
    uses_default_LLM  = input(f"Do you want to use the default LLM {DEFAULT_LLM_MODEL} model? (Y/N): ")
    if uses_default_LLM.lower() in ('y', 'yes'):
        return DEFAULT_LLM_MODEL
    else: 
        custom_LLM = input("See Ollama.com on how to install a local model. Bind your chosen LLM to be used in this script: ")
        return custom_LLM.lower()

def load_llm(chosen_llm) -> BaseLanguageModel:
    llm = OllamaLLM(model=chosen_llm, num_predict=768, num_ctx=4096)
    return llm

def ask_how_many_facts_to_retrive() -> int:
    fact_amount = input("How many facts should the RAG use? 20 is recommended. Only use numbers (0-9): ")
    if fact_amount.isdigit():
        return int(fact_amount)
    else:
        print(f"You used letters (e.g. 'five'). Please retry using only numbers.\n")
        ask_how_many_facts_to_retrive()
    
def will_facts_be_shown_with_answer() -> bool:
    fact_shown = input("Do you want to see all the facts the LLM will use as context before the answer (Y/N): ")
    if fact_shown.lower() in ('y', 'yes'):
        return True
    else: 
        return False

def setup_rag_chain(llm, retriever) -> Runnable:
    prompt = ChatPromptTemplate.from_template("""
    You are an expert enterprise architect chatbot providing information based on the provided ArchiMate model context and knowledge of software architecture.
    Carefully analyze the context and provide a comprehensive and detailed answer to the user's question.
    Synthesize information from multiple parts of the context or your understanding of software architecture if necessary.

    Context:
    {context}

    Question:
    {question}
    """)

    rag_chain = (
        {
            "context": itemgetter("question") | retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            "question": itemgetter("question")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def run_chat_loop(rag, retriever, are_facts_shown) -> None:
    print("\n--- ArchiMate Chatbot Ready ---")
    print("Type your questions, or 'exit' to quit.")
    spinner = Spinner(message="Thinking...", delay=0.1)
    while True:
        question = input("\nYour question: ")
        if question.lower() in ('exit','quit', 'close'):
            print("Exiting chatbot. Goodbye!")
            break
        retrived = retriever.invoke(question)
        context_text = "\n\n".join(doc.page_content for doc in retrived)

        if are_facts_shown:
            print("\n--- Retrived context: ---\n")
            for idx, doc in enumerate(retrived):
                print(f"{idx+1}: {doc.page_content}\n")
                print("---------------------------\n")
        try:
            print("\n--- Answer (this usually takes a few minutes for non-GPU-laptops using 8b-models) ---\n")
            spinner.start()
            response = rag.invoke({"context": context_text, "question": question})
            print(response)
            spinner.stop()
        except Exception as e:
            spinner.stop()
            print(f"An error occurred during response generation: {e}")

def main():
    if should_vector_db_update():
        elements = df_from_XML(et_elements)
        relationships = df_from_XML(et_relationships)
        facts = facts_from_df(elements, relationships)
        document = combine_facts_to_langchain_document(facts)
        vectorstore = vectorise_langchain_document_into_faiss(document, OLLAMA_EMBEDDING_MODEL, FAISS_INDEX_PATH)
    else: 
        vectorstore = load_faiss_index(OLLAMA_EMBEDDING_MODEL, FAISS_INDEX_PATH)
    
    chosen_llm = ask_llm_to_use(DEFAULT_LLM_MODEL)
    llm = load_llm(chosen_llm)
    if llm is None:
        return
    
    k_retrived_amount = ask_how_many_facts_to_retrive()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_retrived_amount})
    are_facts_shown = will_facts_be_shown_with_answer()
    rag = setup_rag_chain(llm, retriever)
    run_chat_loop(rag, retriever, are_facts_shown)

if __name__ == "__main__": 
    main()