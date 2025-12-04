import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.join(BASE_DIR, "..", "policies")

def build_retriever():
    docs = []

    # 정책 폴더 내 모든 txt 파일 로드
    for filename in os.listdir(POLICY_DIR):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(POLICY_DIR, filename), encoding="utf-8")
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(split_docs, embeddings)
    return vectordb.as_retriever()

POLICY_RETRIEVER = build_retriever()