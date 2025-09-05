import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import DirectoryLoader, TextLoader

def create_vs(docs_path, vs_path, model, device):
    embeddings = HuggingFaceEmbeddings(model_name=model, model_kwargs={"device": device})
    if os.path.exists(vs_path):
        return FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)

    loader = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = SemanticChunker(embeddings)
    texts = text_splitter.split_documents(documents)

    vs = FAISS.from_documents(texts, embeddings)
    vs.save_local(vs_path)
    return vs
