import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone


load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')




embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)


# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "fuse-chat-project"


def split_text(text):
    docs = text_splitter.create_documents([text])
    return docs

def all_emb(chunks):
    docsearch = Pinecone.from_texts([t.page_content for t in chunks], embeddings, index_name=index_name)
    return docsearch


def chat_module(ask, query):
    
    answer = ask.similarity_search(query, k = 1)

    ans = str(answer)

    return ans
