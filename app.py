from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DATA_PATH = "data/"
#DB_FAISS_PATH = "vectorstores/db_faiss"


# create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = FAISS.from_documents(texts, embeddings)
    return(vectorstore)




prompt_template = """Human: Use the following pieces of context to provide a concise answer to
the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Assistant:"""
prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)



def create_RetrievalQA_chain(query):
    print("Connecting to bedrock")
    vector_store = create_vector_db()
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 5, "include_metadata": True}
    )
    chain = RetrievalQA.from_chain_type(
        
       # llm=bedrock_llm,
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",
                     model_kwargs={"temperature":0.5, "max_length":512}),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    result = chain({"query": query})
    return result

user_input = st.text_area("Enter Text To summarize")
button = st.button("Generate Summary")
if user_input and button:
    summary = create_RetrievalQA_chain(user_input)
    st.write("Summary : ", summary)
