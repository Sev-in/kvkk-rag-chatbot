# 📄 KVKK RAG Chatbot

A production-ready, Retrieval-Augmented Generation (**RAG**) chatbot built for **KVKK (Kişisel Verilerin Korunması Kanunu)** compliance and education.  
It indexes official KVKK documents, retrieves the most relevant sections using **Chroma**, and generates accurate, context-aware answers with **Google Gemini**.

---

## 🚀 Features

- Simple RAG pipeline: `chunk → embed → store → retrieve → generate`
- **Google Gemini 2.0 Flash** for fast and reliable generation
- **Hugging Face multilingual MiniLM** for high-quality Turkish embeddings
- **Streamlit UI** with persistent conversation history
- **Chroma vector store** for efficient semantic search
- Context-grounded answers: never hallucinates outside the provided data

---

## 🧠 Tech Stack

| Layer             | Technology                                                                   |
| ----------------- | ---------------------------------------------------------------------------- |
| **Frontend**      | Streamlit                                                                    |
| **Backend / RAG** | LangChain + Chroma                                                           |
| **LLM**           | Google Gemini (`gemini-2.0-flash-exp`)                                       |
| **Embeddings**    | Hugging Face – `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **Data Source**   | Official KVKK PDFs (`kvkk_veri_seti/`)                                       |
| **Environment**   | Python 3.10+, dotenv                                                         |

---

## ⚙️ Requirements

- Python **3.10+**
- A valid **Google API key** with access to **Gemini models**

---

## 🧩 Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/kvkk-rag-chatbot.git
cd kvkk-rag-chatbot
```

### 2️⃣ Install dependencies

pip install -r requirements.txt

### 3️⃣ Configure environment

Create a .env file in the project root:
GOOGLE_API_KEY=your_google_api_key_here

### 4️⃣ Create the vector database

(Only required the first time or after data/model changes)
python veri_isleme.py

### 5️⃣ Start the app

streamlit run app.py

kvkk-rag-chatbot/
├── kvkk_veri_seti/ # PDF dataset (official KVKK documents)
├── chroma_db_kvkk/ # Persisted vector database (auto-created)
├── app.py # Streamlit app + RAG pipeline
├── veri_isleme.py # PDF processing + embedding + storage
├── requirements.txt # Dependencies
├── .env # GOOGLE_API_KEY (not committed)
└── README.md # Project documentation
