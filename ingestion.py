import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Environment variables (.env) లోడ్ చేయడానికి
load_dotenv()

pdf_path = r"D:\HEMA PROJECTS\DocuSense-AI\Data\Premium Series Maths Multiple Choice 4 Sets Pack A.pdf"

def load_split_and_save_pdf(path):
    # 1. PDF లోడ్ చేయడం
    loader = PyPDFLoader(path)
    documents = loader.load()
    
    # 2. Text Splitting (ముక్కలు చేయడం)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Success! PDF loaded and split into {len(chunks)} chunks.")
    
    # 3. Embeddings Setup
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 4. Vector DB లోకి సేవ్ చేయడం (ఇది చాలా ముఖ్యం బ్రో!)
    print("డేటాబేస్ లోకి ముక్కలను సేవ్ చేస్తున్నాను, దయచేసి వెయిట్ చేయండి...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print("🎉 అద్భుతం! 'chroma_db' ఫోల్డర్ లోకి డేటా సక్సెస్‌ఫుల్‌గా సేవ్ అయిపోయింది బ్రో!")
    return vector_db

if __name__ == "__main__":
    if os.path.exists(pdf_path):
        load_split_and_save_pdf(pdf_path)
    else:
        print(f"Error: ఈ పాత్ లో PDF దొరకలేదు బ్రో! -> {pdf_path}")
