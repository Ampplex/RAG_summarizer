import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(pages)

def embed_and_store_chunks(docs):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embedding=embed_model, persist_directory="rag_db")
    vectordb.persist()
    return vectordb

def retrieve_context(db, query, k=5):
    retriever = db.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join(doc.page_content for doc in docs)

def summarize_with_gemini(context, query="Summarize this PDF"):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f""" You are an expert in Natural Language Processing. Summarize the following content clearly and briefly.
Context:
{context}

Question: {query}
Answer:
"""
    response = model.generate_content(prompt)
    return response.text

def run_pdf_rag(pdf_path):
    print("Loading PDF...")
    docs = load_and_split_pdf(pdf_path)

    print("Creating vector store...")
    vectordb = embed_and_store_chunks(docs)

    print("Retrieving relevant chunks...")
    context = retrieve_context(vectordb, "Summarize this PDF")

    print("Generating summary with Gemini...")
    summary = summarize_with_gemini(context)
    print("\n Summary:\n")
    print(summary)


if __name__ == "__main__":
    run_pdf_rag("attention_is_all_you_need.pdf")  # Replace with your file