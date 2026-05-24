# 💻 DocuSense-AI: Smart Math Tutor

An advanced, high-performance **Multimodal AI Math Tutor** built using **Streamlit** and **Google Gemini API**. This application provides instantaneous, step-by-step logical explanations for text-based math equations as well as complex mathematical diagrams pasted directly from the clipboard.

## 🚀 Live Demo
Experience the live application here: [DocuSense-AI Live App](https://docusense-ai-btsdw2vc94dev6qtbi9pom.streamlit.app/)

---

## ✨ Features

- **Multimodal Capabilities:** Processes both textual math queries and visual diagrams or screenshots.
- **Instant Clipboard Parsing:** Features seamless `streamlit-paste-button` integration—just take a screenshot with your Snipping Tool and paste it directly into the app.
- **Real-Time Streaming:** Leveraging Gemini's streaming interface to deliver dynamic, typing-effect solutions instantly.
- **Responsive Architecture:** Fully optimized UI/UX that scales beautifully from high-resolution desktop monitors down to mobile browsers.
- **RAG Infrastructure Ready:** Includes scalable backend logic (`ingestion.py`, `vector_store.py`) prepared for vector database expansions using LangChain and ChromaDB.

---

## 🛠️ Tech Stack

- **Frontend & UI:** Streamlit, Pillow (PIL), HTML/CSS Custom Styling
- **AI Ecosystem:** Google Gemini API (`gemini-2.5-flash`), LangChain Framework
- **Data & Storage:** ChromaDB (Vector Store), Protobuf
- **Environment Management:** Python 3.12, Python-dotenv

---

## 💻 Local Installation & Setup

Follow these steps to set up and run the project locally:

1. **Clone the Repository:**
```bash
   git clone [https://github.com/hemalathakadapa/DocuSense-AI.git](https://github.com/hemalathakadapa/DocuSense-AI.git)
   cd DocuSense-AI
