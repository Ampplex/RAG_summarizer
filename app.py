import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(pages)

def create_vectordb(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embedding=embeddings)
    return vectordb

def get_context(vectordb, query="summarize"):
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context[:3000]

def summarize(context):
    model_name = "microsoft/DialoGPT-medium"  # Better than TinyLlama
    # Alternative: "facebook/bart-large-cnn" (but larger)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    prompt = f"Summarize in 2-3 sentences: {context[:1000]}\n\nSummary:"
    
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.2
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new part
    if "Summary:" in result:
        summary = result.split("Summary:")[-1].strip()
    else:
        summary = result[len(prompt):].strip()
    
    return summary if summary else "Could not generate summary"

def main():
    pdf_path = "attention_is_all_you_need.pdf"  # Change this
    
    print("Loading PDF...")
    docs = load_pdf(pdf_path)
    
    print("Creating vector database...")
    vectordb = create_vectordb(docs)
    
    print("Getting context...")
    context = get_context(vectordb)
    
    print("Summarizing...")
    summary = summarize(context)
    
    print("\nSummary:")
    print(summary)

if __name__ == "__main__":
    main()