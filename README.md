# 📚 Chat with Multiple PDFs (RAG App)

A **Retrieval-Augmented Generation (RAG)** web application that allows users to upload **multiple PDF documents** and chat with them using a local LLM powered by **Ollama** and **LangChain**.

Built with **Streamlit**, this app extracts text from PDFs, embeds them into a vector database, and retrieves relevant context to answer user questions accurately.

---

## ✨ Features

* 📄 Upload **multiple PDFs** at once
* 🔍 Semantic search using **vector embeddings** (FAISS)
* 💬 Conversational chat interface with **user & bot bubbles**
* 🧠 Local LLM inference using **Ollama (Llama 3)**
* ⚡ Fast responses with chunking & retrieval optimization
* 📊 Sidebar status updates (e.g. *Processing complete*)

---

## 🏗️ Tech Stack

* **Python 3.10+**
* **Streamlit** – Web UI
* **LangChain** – RAG pipeline & memory
* **Ollama** – Local LLM (Llama 3)
* **FAISS** – Vector similarity search
* **HuggingFace Embeddings** – Text embeddings
* **PyPDF2** – PDF text extraction

---

## 📂 Project Structure

```
multiple-pdf-chat/
│── app.py                 # Main Streamlit app
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
│── .gitignore             # Ignored files (venv, .env, etc.)
│── .env                   # API keys (NOT committed)
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/multiple-pdf-chat.git
cd multiple-pdf-chat
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS / Linux
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Ollama Setup (Required)

1. Install Ollama: [https://ollama.com](https://ollama.com)
2. Pull the Llama 3 model:

```bash
ollama pull llama3
```

3. Make sure Ollama is running:

```bash
ollama run llama3
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

> ⚠️ **Never commit `.env` files to GitHub**

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 🧠 How It Works (RAG Flow)

1. User uploads PDFs
2. Text is extracted and chunked
3. Chunks are embedded and stored in FAISS
4. User asks a question
5. Relevant chunks are retrieved
6. LLM generates an answer using retrieved context

---

## 🚀 Future Improvements

* 🔄 Persistent vector store (disk-based FAISS)
* 🗂️ PDF source citations in answers
* 🌐 Cloud deployment (Streamlit Cloud)
* 🧩 Support for more file formats
* 🔍 Advanced retriever tuning

---

## 🧑‍💻 Author

Built by **YiQi Xiang**
🎓 University of Waterloo – Statistics / Computer Science
💼 Interests: Data, AI, ML, Backend Systems

---

## ⭐ Acknowledgements

* LangChain Documentation
* Ollama Community
* HuggingFace Transformers

---

If you find this project useful, feel free to ⭐ the repo!
