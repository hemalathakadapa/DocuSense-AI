import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

print("\n--- Running Query ---")

load_dotenv()

# 1. Embeddings & Vector DB Setup
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# --- DEBUG CODE ---
try:
    collection_count = vector_db._collection.count()
    print(f"[DEBUG] Database లో ఉన్న మొత్తం ముక్కల సంఖ్య (Chunks): {collection_count}")
except Exception as e:
    print("[DEBUG] Database కనెక్ట్ అవ్వలేదు:", str(e))
# ------------------

# 2. Retriever & LLM Setup
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 3. Prompt Design
prompt = ChatPromptTemplate.from_template("""
You are an expert Math tutor. Based ONLY on the provided context from the document, 
solve the user's question step-by-step with clear explanations and formulas.

Context:
{context}

Question: {input}
Answer:""")

# 4. RAG Chain Setup
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Query
query = "Find 2 multiple choice questions related to Algebra from the document and provide step-by-step solutions."


response = rag_chain.invoke(query)

print("\n--- Answer ---")
print(response)
