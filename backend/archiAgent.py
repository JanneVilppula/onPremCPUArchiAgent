from typing import List, Dict, Optional
from langchain_core.language_models.base import BaseLanguageModel, Runnable

import sys
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


OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "deepseek-r1"
FAISS_INDEX_PATH = "facts_faiss_index"
try:
    ARCHI_XML_PATH = max(list(Path(__file__).parent.glob('*.xml')), key=lambda p: p.stat().st_mtime)
    tree = et.parse(ARCHI_XML_PATH)
    root = tree.getroot()
    et_elements = root[1]
    et_relationships = root[2]
    ARCHIMATE_NAMESPACE = "{http://www.opengroup.org/xsd/archimate/3.0/}"
    XSI_NAMESPACE = "{http://www.w3.org/2001/XMLSchema-instance}"
except:
    print(f"Cannot find Archi-export XML. Ensure the python-script is in the same folder with the sole XML-file in that folder.")

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
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(document, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    loaded_vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("FAISS index loaded")
    return loaded_vectorstore

def load_faiss_index(OLLAMA_EMBEDDING_MODEL, FAISS_INDEX_PATH) -> Optional[FAISS]:
    return

def ask_llm_to_use(DEFAULT_LLM_MODEL) -> Optional[str]:
    return

def load_llm(chosen_llm) -> Optional[BaseLanguageModel]:
    return

def ask_how_many_facts_to_retrive() -> int:
    return

def will_facts_be_shown_with_answer() -> bool:
    return

def get_facts_for_rag() -> List[Dict[str, str]]:
    return

def create_rag_prompt() -> ChatPromptTemplate:
    return

def setup_rag_chain(llm, vectorstore, prompt, k_retrived_amount) -> Runnable:
    return

def run_chat_loop(rag, are_facts_shown) -> None:
    return

def main():
    print("main started")
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
    
    prompt = create_rag_prompt()
    k_retrived_amount = ask_how_many_facts_to_retrive()
    are_facts_shown = will_facts_be_shown_with_answer()
    rag = setup_rag_chain(llm, vectorstore, prompt, k_retrived_amount)
    run_chat_loop(rag, are_facts_shown)

if __name__ == "__main__": 
    main()