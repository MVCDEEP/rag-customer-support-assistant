from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_db"

def get_retriever(k=3):
    from langchain_community.embeddings import FakeEmbeddings
    embeddings = FakeEmbeddings(size=384)
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})

def retrieve_chunks(query: str, k=3):
    retriever = get_retriever(k)
    docs = retriever.invoke(query)
    return docs