import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from summarize import summarizer


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
                chunk_size = 1000,
                chunk_overlap = 200,
                length_function = len
            )

            chunks = text_splitter.split_text(text)
            
            intermediate_output, time_first = summarizer(chunks)

            new_chunks = text_splitter.split_text(intermediate_output)

            output, time_second = summarizer(new_chunks)


            st.write(output)
            st.write(f"Total time taken for summary generation is {time_first + time_second} seconds.")
    
    else:
        text_input = st.sidebar.text_input(
            "Enter your arxiv Link here 👇",
            # label_visibility='visible',
            placeholder='Arvix Link',
        )
    
    text_input = st.text_input(
                "Ask your paper here and hit enter 👇",
                # label_visibility='visible',
                placeholder='',
                )
    
    if text_input:
        results = text_input
        st.write(results)
        
    

if __name__ == '__main__':
    main()