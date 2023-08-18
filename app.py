import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from summarize import summarizer, abstractive_sum
from chat import chat_module, split_text, all_emb
import torch
import os
from dotenv import load_dotenv
load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"

pytorch_cuda_alloc_conf = os.getenv("PYTORCH_CUDA_ALLOC_CONF")

def main():
    st.sidebar.title("Interact with Paper 📚")

    source = ("PDF", "ARXIV LINK")
    source_index = st.sidebar.selectbox("Select Input type", range(
        len(source)), format_func=lambda x: source[x])
    
    if source_index == 0:
        pdf = st.sidebar.file_uploader(
            "Load File", type=['pdf'])
        # Extract the text of pdf
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size = 3000,
                chunk_overlap = 300,
                length_function = len
            )

            torch.cuda.empty_cache()

            chunks = text_splitter.split_text(text)
            
            output, first_time = summarizer(chunks)

            time = first_time 

            st.write(output)
            st.write(f"Total time taken for summary generation is {time} seconds.")
    
    else:
        text_input = st.sidebar.text_input(
            "Enter your arxiv Link here 👇",
            # label_visibility='visible',
            placeholder='Arvix Link',
        )
    
    text_input = st.text_input(
                "Ask your paper here and hit enter 👇",
                # label_visibility='visible',3
                placeholder='Your Query',
                )
    
    if text_input:
        chat_chunks = split_text(text)
        ask = all_emb(chat_chunks)
        op = chat_module(ask, text_input)
        st.write(op)
        
    

if __name__ == '__main__':
    main()