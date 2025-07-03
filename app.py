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
    return context[:3000]  # Keep it short

def summarize(context):
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    prompt = f"""<|system|>
You are a helpful assistant that writes clear, concise summaries. Write in complete sentences about the main ideas.
<|user|>
Please summarize the key points from this text in 2-3 sentences:

{context}
<|assistant|>
The main points are: """
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1500, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200, 
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in result:
        summary = result.split("<|assistant|>")[-1].strip()
    else:
        summary = result.split("The main points are:")[-1].strip()
    
    return summary

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