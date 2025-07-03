from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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

# Retrieve top-k relevant chunks, truncate to safe size
def retrieve_context(db, query, k=5, max_chars=3000):
    retriever = db.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    combined = "\n\n".join(doc.page_content for doc in docs)
    return combined[:max_chars]  # truncate context safely

# Local summarizer using TinyLlama
def summarize_locally(context, query="Summarize this PDF"):
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    prompt = f"""Below is a passage from a research paper. Summarize it briefly and clearly.

Passage:
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

    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Summary:")[-1].strip()

# Main flow
def run_pdf_rag(pdf_path):
    print("Loading PDF...")
    docs = load_and_split_pdf(pdf_path)

    print("Creating vector store...")
    vectordb = embed_and_store_chunks(docs)

    print("Retrieving relevant chunks...")
    context = retrieve_context(vectordb, "Summarize this PDF")

    print("Generating summary locally...")
    summary = summarize_locally(context)

    print("\nSummary:\n")
    print(summary)

if __name__ == "__main__":
    run_pdf_rag("attention_is_all_you_need.pdf")