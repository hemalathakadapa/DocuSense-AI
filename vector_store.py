import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from ingestion import load_and_split_pdf  # ingestion.py నుండి ఫంక్షన్ ని ఇంపోర్ట్ చేసుకుంటున్నాం

# .env నుండి కీ ని లోడ్ చెయ్యి
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def create_vector_db():
    # 1. డేటాని లోడ్ చేసి స్ప్లిట్ చెయ్యి
    pdf_path = "data/sample.pdf"
    chunks = load_and_split_pdf(pdf_path) # ఇందాకటి ఫంక్షన్ ని వాడుతున్నాం

    # 2. Embedding Model (Google Gemini)
    # పాతది తీసేసి, ఇలా పెట్టు:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


    
    
   
   

    # 3. Vector Database లో సేవ్ చెయ్యి
    # db అనే ఫోల్డర్ లో ఈ డేటా సేవ్ అవుతుంది
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("Success! Vector Database created and saved in './chroma_db'.")

if __name__ == "__main__":
    create_vector_db()