
import os
import streamlit as str_ui
from dotenv import load_dotenv
from PIL import Image
import io  
import base64  
from streamlit_paste_button import paste_image_button  
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage

# Load environment variables from the .env file (e.g., GOOGLE_API_KEY)
load_dotenv()

# Configure the Streamlit web page title, icon, and wide layout
str_ui.set_page_config(page_title="DocuSense-AI", page_icon="📚", layout="wide")

# --- FIXED: CUSTOM CSS TO CHANGE UI FONT TO ROBOTO/SANS-SERIF ---
str_ui.markdown(
    """
    <style>
    /* Change font for the entire app body, input fields, and buttons */
    html, body, [class*="css"], .stTextInput, .stButton, p, span {
        font-family: 'Roboto', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    }
    /* Style headers specifically for a sharper look */
    h1, h2, h3 {
        font-family: 'Roboto', 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True  # Fixed the unexpected argument error here bro!
)

str_ui.title("📚 DocuSense-AI: Smart Math Tutor")
str_ui.write("Ask questions from the database, or paste a math diagram (image) from your Snipping Tool directly!")

# Cache the vector database resource so it doesn't reload on every user interaction
@str_ui.cache_resource
def get_vector_db():
    # Initialize the same Google Embedding model used during ingestion
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    # Load the local Chroma database directory
    return Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Initialize the vector database, retriever (fetching top 5 matches), and the LLM
vector_db = get_vector_db()
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Added temperature=0 to make the model more deterministic and faster
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Divide the UI layout into two equal columns: Left for input, Right for output
col1, col2 = str_ui.columns([1, 1])

# --- LEFT COLUMN: INPUT SECTION ---
with col1:
    str_ui.subheader("📂 Input Section")
    
    # Text input box for the user to type or copy-paste their text question
    user_query = str_ui.text_input("Type your question here (Can copy-paste text):", placeholder="e.g., Solve the question in this image.")
    
    str_ui.write("📸 **Math Diagram Paste Section:**")
    str_ui.caption("Take a screenshot using Snipping Tool, then click the button below to paste!")
    
    # Render the special clipboard paste button
    paste_result = paste_image_button(
        label="📋 Paste Image from Clipboard"
    )
    
    image = None
    # If the user successfully pasted an image, display it on the screen
    if paste_result.image_data is not None:
        image = paste_result.image_data
        str_ui.image(image, caption="💥 Success! Screenshot pasted.", width=300)

    # Action button to trigger the execution pipeline
    submit_button = str_ui.button("Solve Now 🚀")

# --- RIGHT COLUMN: OUTPUT SECTION ---
with col2:
    str_ui.subheader("📝 Solution:")
    
    if submit_button:
        # Check if the user clicked the button without providing any text or image
        if not user_query and image is None:
            str_ui.warning("Please type a question or paste an image first, bro!")
        else:
            with str_ui.spinner("Our Smart Math Tutor is thinking..."):
                try:
                    # CASE 1: If an image is pasted, use Gemini's Vision capabilities with streaming
                    if image is not None:
                        # Convert PIL Image to bytes
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        
                        # Convert bytes to Base64 String
                        base64_image = base64.b64encode(img_bytes).decode('utf-8')
                        
                        # Enforcing minimum steps and concise response in fallback prompt
                        prompt_msg = user_query if user_query else "Solve the math problem shown in this image using the MINIMUM number of steps required. Keep the explanation very brief and highlight the final answer."
                        if user_query:
                            prompt_msg += " (Provide the solution in minimum steps and keep it highly concise)."
                        
                        # Format the multimodal input message properly using Base64 Data URL
                        message = HumanMessage(
                            content=[
                                {"type": "text", "text": prompt_msg},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                                }
                            ]
                        )
                        
                        # Using .stream() to get response chunk by chunk for ultra-fast experience
                        response_stream = llm.stream([message])
                        
                        # Helper generator function to stream content cleanly into the UI
                        def stream_generator(stream):
                            for chunk in stream:
                                yield chunk.content

                        str_ui.write_stream(stream_generator(response_stream))
                    
                    # CASE 2: If it's a text-only query, run the standard RAG pipeline from local DB
                    else:
                        # Custom prompt edited to force "MINIMUM steps" and "highly concise explanations"
                        prompt = ChatPromptTemplate.from_template("""
                        You are an expert Math tutor. Based ONLY on the provided context from the document, 
                        solve the user's question using the MINIMUM number of steps required. 
                        Keep the explanations highly concise, brief, and directly to the point. Do not write unnecessary sentences.

                        Context:
                        {context}

                        Question: {input}
                        Answer:""")

                        # Helper function to format and combine the retrieved document chunks
                        def format_docs(docs):
                            return "\n\n".join(doc.page_content for doc in docs)

                        # Assemble the LangChain RAG pipeline chain with streaming support
                        rag_chain = (
                            {"context": retriever | format_docs, "input": RunnablePassthrough()}
                            | prompt
                            | llm
                            | StrOutputParser()
                        )
                        
                        # Streaming text-only RAG response
                        str_ui.write_stream(rag_chain.stream(user_query))
                    
                except Exception as e:
                    # Catch and display any runtime errors cleanly in the UI
                    str_ui.error(f"Got a small error bro: {str(e)}")