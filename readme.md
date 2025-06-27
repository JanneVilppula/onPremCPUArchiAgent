# on-prem CPU archiAgent RAG-bot

archiAgent.py will run on the command line and convert an Archi xml-export into a FAISS vector database that a locally installed Ollama LLM model can RAG-query as context to your questions.

## How to install

0. Ensure you have archiAgent.py and your Archi XML-export in the same folder without any other XML-files in it.
1. `pip install requirements.txt` in command line to get all dependencies for the script
2. `python archiAgent.py`  in command line to run the script
3. First time running, create the FAISS vector database
4. Ensure you have Ollama and an LLM model installed locally to use with archiAgent

## How to use

1. In Archi File > Export > Model to Open Exchange File to export the XML
	1. [Archi XMLs online](https://github.com/archimatetool/ArchiModels) are often very old and incompatible format to the script. If you download them, get the .archimate-file and make the XML-export in Archi
	2. The Archi file must have a name and documentation to work!
	3. The more documentation you write for the processes and relationships (often skipped), the better the facts!
![[Pasted image 20250627135557.png]]
2. Only update the vector database if you have updated the XML file
3. The mistral:7b model will answer in around a minute with laptops without GPUs. More open-ended questions benefit from deepseek-r1:8b, with around 3-6 minute answer time
4. The more facts you select for context, the slower the answer will be. Over 20 facts appeared to not improve context much.
![[Pasted image 20250627140124.png]]
5. The context window and answer length have been choked to speed answer times, so the chatbot won't remember the ongoing discussion.
![[Pasted image 20250627140515.png]]
6. Write `exit` to close chatbot
![[Pasted image 20250627140811.png]]
7. Try different models and fact amounts to get different answers
![[Pasted image 20250627141354.png]]
*deepseek's "thinking" mode ate up the entire answer length so the answer is just reasoning about the question with it* 
8. Fill in your relationship and business process descriptions in Archi to get better facts
![[Pasted image 20250627141826.png]]
*the demo "logyCorp" has all its relationships and business processes documented*
![[Pasted image 20250627141938.png]]
![[Pasted image 20250627142107.png]]
*the demo "metalArchi" doesn't have documentation for its relationships and business processes*
![[Pasted image 20250627142219.png]]

## How it works
1. Selects the most recent XML file in the folder and accesses it's elements and relationships (which are flattened from all views, so no view-specific context, i.e., where relationships are)
```python
ARCHI_XML_PATH = max(list(Path(__file__).parent.glob('*.xml')), key=lambda p: p.stat().st_mtime)
tree = et.parse(ARCHI_XML_PATH)
root = tree.getroot()
et_elements = root[2]
et_relationships = root[3]
```
2. Parses the XML as a pandas dataframe
```python
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
    
def main():
    if should_vector_db_update():
        elements = df_from_XML(et_elements)
        relationships = df_from_XML(et_relationships)
        facts = facts_from_df(elements, relationships)
        document = combine_facts_to_langchain_document(facts)
        vectorstore = vectorise_langchain_document_into_faiss(document, OLLAMA_EMBEDDING_MODEL, FAISS_INDEX_PATH)
    else:
        vectorstore = load_faiss_index(OLLAMA_EMBEDDING_MODEL, FAISS_INDEX_PATH)
```
3. Writes "fact sentences" from the relationships and elements
	1. I've opted for "*source* (*type*) is the *source* of a *relationship* to *target* (*type*)" syntax to keep facts short.
	2. I also experimented with longer facts of "*source (type)* has the following relationships: *relationship* to/from *target*" but this made very long facts where most of the relationship information was not pertinent to question asked 
```python
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
```
4. Combines "fact sentences" into langchain document to vectorise into a FAISS database
```python
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
```
5. Asks preferences and applies them
```python
def ask_llm_to_use(DEFAULT_LLM_MODEL) -> str:
    uses_default_LLM  = input(f"Do you want to use the default LLM {DEFAULT_LLM_MODEL} model? (Y/N): ")
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
```
6. Sets up RAG pipeline
```python
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

def main():
    k_retrived_amount = ask_how_many_facts_to_retrive()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_retrived_amount})
    are_facts_shown = will_facts_be_shown_with_answer()
    rag = setup_rag_chain(llm, retriever)
    run_chat_loop(rag, retriever, are_facts_shown)
```
7. Runs chat loop until `exit` from program
```python
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
```

