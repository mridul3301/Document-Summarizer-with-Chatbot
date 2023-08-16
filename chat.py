import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")


from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

from langchain.vectorstores import FAISS


from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)



def split_text(text):
    docs = text_splitter.create_documents([text])
    return docs


def chat_module(text, query):
    db = FAISS.from_documents(text, embeddings)

    question = query
    answer = db.similarity_search(question)

    ans = str(answer)

    return ans
