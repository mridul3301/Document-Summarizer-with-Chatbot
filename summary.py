import streamlit as st
import fitz  # PyMuPDF
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import os

# Load Pegasus model and tokenizer
model_name = "google/pegasus-xsum"
model = PegasusForConditionalGeneration.from_pretrained(model_name)
tokenizer = PegasusTokenizer.from_pretrained(model_name)
max_input = 1024
max_summary = 50

def generate_long_text_summary(long_text, max_segment_length=800):
    segments = [long_text[i:i + max_segment_length] for i in range(0, len(long_text), max_segment_length)]
    summaries = []

    for segment in segments:
        summary = generate_summary(segment)
        summaries.append(summary)

    overall_summary = " ".join(summaries)
    return overall_summary

# ...

def main():
    st.title("Research Paper Summarizer")
    st.write("Choose one option: Upload a PDF or Input Text")

    option = st.radio("Select an option", ("Upload PDF", "Input Text"))

    if option == "Upload PDF":
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

        if uploaded_file is not None:
            st.subheader("Original Paper")
            pdf_text = extract_pdf_text(uploaded_file)
            st.write(pdf_text)

            if st.button("Generate Summary"):
                if len(pdf_text) > max_input:
                    summary = generate_long_text_summary(pdf_text)
                else:
                    summary = generate_summary(pdf_text)
                st.subheader("Generated Summary")
                st.write(summary)

    elif option == "Input Text":
        input_text = st.text_area("Input Text", height=200)

        if input_text:
            if st.button("Generate Summary"):
                if len(input_text) > max_input:
                    summary = generate_long_text_summary(input_text)
                else:
                    summary = generate_summary(input_text)
                st.subheader("Generated Summary")
                st.write(summary)

def extract_pdf_text(pdf_file):
    # Save the uploaded PDF locally
    with open("temp.pdf", "wb") as temp_pdf:
        temp_pdf.write(pdf_file.read())

    # Open the saved PDF
    pdf_document = fitz.open("temp.pdf")

    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    pdf_document.close()

    # Clean up by deleting the temporary file
    os.remove("temp.pdf")

    return text


def generate_summary(text):
    inputs = tokenizer(text, max_length=max_input , return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length= max_summary, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    main()
