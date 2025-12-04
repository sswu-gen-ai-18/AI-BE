import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.join(BASE_DIR, "..", "policies")

def build_retriever():
    # 1) 문서 로드
    docs = []
    for filename in os.listdir(POLICY_DIR):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(POLICY_DIR, filename), encoding="utf-8")
            docs.extend(loader.load())

    # 2) chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    split_docs = splitter.split_documents(docs)

    # 3) Embeddings
    embeddings = OpenAIEmbeddings()

    # 4) Vector DB: FAISS 사용
    vectordb = FAISS.from_documents(split_docs, embeddings)

    return vectordb.as_retriever()

POLICY_RETRIEVER = build_retriever()
