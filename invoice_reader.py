import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline

st.set_page_config(page_title="Invoice AI Reader", layout="centered")

st.title("📄 AI Invoice Reader")
st.write("Upload a PDF invoice and ask questions. The AI will answer based on the document.")

# File uploader
uploaded_file = st.file_uploader("Upload Invoice PDF", type="pdf")

if uploaded_file:
    # Read PDF content
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()

    st.success("✅ Invoice uploaded and read successfully!")

    # Question input
    question = st.text_input("Ask a question about the invoice")

    if question:
        # Hugging Face pipeline
        qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        result = qa(question=question, context=text)
        st.markdown(f"### 🤖 Answer: {result['answer']}")
