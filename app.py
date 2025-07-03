from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# Load and split PDF
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(pages)

# Embed and store chunks in vector DB
def embed_and_store_chunks(docs):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embedding=embed_model, persist_directory="rag_db")
    vectordb.persist()
    return vectordb

# Retrieve top-k relevant chunks
def retrieve_chunks(db, query, k=5):
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(query)

# Post-process summary to reduce repetition
def clean_summary(summary: str) -> str:
    summary = re.sub(r'\b(The|This) research paper\b', '', summary, flags=re.IGNORECASE)
    summary = re.sub(r'\s+', ' ', summary).strip()
    return summary

# Load model and tokenizer once
def load_local_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return model, tokenizer

# Summarize a single chunk
def summarize_chunk(text, model, tokenizer):
    prompt = f"""You are an expert in NLP. Summarize the following technical passage clearly and briefly:

{text}

Summary:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = decoded.split("Summary:")[-1].strip()
    return clean_summary(summary)

# Reduce step – summarize combined summaries
def summarize_summaries(summaries, model, tokenizer):
    joined = "\n".join(f"- {s}" for s in summaries)
    prompt = f"""Combine the following bullet-point summaries into one concise and cohesive summary of the paper:

{joined}

Final Summary:"""

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

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_summary = decoded.split("Final Summary:")[-1].strip()
    return clean_summary(final_summary)

# Main flow
def run_pdf_rag(pdf_path):
    print("Loading PDF...")
    docs = load_and_split_pdf(pdf_path)

    print("Creating vector store...")
    vectordb = embed_and_store_chunks(docs)

    print("Retrieving relevant chunks...")
    relevant_docs = retrieve_chunks(vectordb, "Summarize this PDF", k=5)

    print("Loading model...")
    model, tokenizer = load_local_model()

    print("Summarizing each chunk (map step)...")
    summaries = []
    for doc in relevant_docs:
        chunk_summary = summarize_chunk(doc.page_content[:1500], model, tokenizer)
        summaries.append(chunk_summary)

    print("Reducing summaries into final summary...")
    final_summary = summarize_summaries(summaries, model, tokenizer)

    print("\nSummary:\n")
    print(final_summary)

if __name__ == "__main__":
    run_pdf_rag("attention_is_all_you_need.pdf")  # Replace with your file