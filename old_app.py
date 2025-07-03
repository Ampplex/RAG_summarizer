import os
import torch
import re
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load and split PDF into chunks
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(pages)

# Create vector database
def embed_and_store_chunks(docs):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embedding=embed_model, persist_directory="rag_db")
    vectordb.persist()
    return vectordb

# Retrieve relevant chunks based on a query
def retrieve_context(db, query, k=5, max_chars=3000):
    retriever = db.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    combined = "\n\n".join(doc.page_content for doc in docs)
    return combined[:max_chars]  # truncate safely for local model

# Clean repetition in the final summary
def clean_summary(summary: str) -> str:
    summary = re.sub(r'\b(The|This) research paper\b', '', summary, flags=re.IGNORECASE)
    summary = re.sub(r'\s+', ' ', summary).strip()
    return summary

# Local summarizer using TinyLlama
def summarize_with_local_llm(context):
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    prompt = f"""You are an expert in NLP. Summarize the key ideas from the following text clearly and briefly.

Context:
{context}

Summary:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )

    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = raw_output.split("Summary:")[-1].strip()
    return clean_summary(summary)

# Main execution
def run_pdf_rag(pdf_path):
    print("Loading PDF...")
    docs = load_and_split_pdf(pdf_path)

    print("Creating vector store...")
    vectordb = embed_and_store_chunks(docs)

    print("Retrieving relevant chunks...")
    context = retrieve_context(vectordb, "Summarize this PDF")

    print("Generating summary locally...")
    summary = summarize_with_local_llm(context)

    print("\nSummary:\n")
    print(summary)

if __name__ == "__main__":
    run_pdf_rag("attention_is_all_you_need.pdf")  # Replace with your file