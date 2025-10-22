# app.py
import streamlit as st
import os
from dotenv import load_dotenv

# --- LangChain ve Google GenerativeAI importları ---
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# ======================================================
# ===============  API ANAHTARI YÖNETİMİ  ===============
# ======================================================

def get_api_key() -> str:
    """GOOGLE_API_KEY'i .env dosyasından alır."""
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        st.error("❌ GOOGLE_API_KEY ortam değişkeni bulunamadı. Lütfen `.env` dosyanızı kontrol edin.")
        st.stop()
    return key


def configure_genai(api_key: str):
    """Google GenAI istemcisini yapılandırır."""
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"Google GenAI yapılandırılırken hata oluştu: {e}")
        st.stop()


# ======================================================
# ================  RAG ZİNCİRİ YÜKLEME  ===============
# ======================================================

@st.cache_resource(show_spinner=True)
def load_rag_chain():
    """
    Diskteki Chroma DB'den retriever oluşturur,
    Google Embeddings ve Gemini LLM ile RAG zincirini kurar.
    """
    st.info("RAG zinciri yükleniyor... Lütfen bekleyin.")

    # 1. Gemini embedding modeli
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        st.error(f"Embedding modeli yüklenemedi: {e}")
        st.stop()

    # 2. Kayıtlı Chroma DB'yi yükle
    persist_directory = "./chroma_db_kvkk"
    if not os.path.exists(persist_directory):
        st.error(f"Chroma veritabanı bulunamadı: {persist_directory}")
        st.error("Lütfen önce `create_chroma_gemini.py` dosyasını çalıştırarak veritabanını oluşturun.")
        st.stop()

    try:
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    except Exception as e:
        st.error(f"Chroma yüklenirken hata oluştu: {e}")
        st.stop()

    # 3. Google Gemini LLM
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",  # hızlı ve uygun maliyetli model
            temperature=0.55
        )
    except Exception as e:
        st.error(f"LLM yüklenemedi: {e}")
        st.stop()

    # 4. Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 8})

    # 5. Sistem prompt’u
    system_prompt = (
        "Sen, Kişisel Verilerin Korunması Kanunu (KVKK) konusunda uzman bir yapay zekasın. "
        "Cevaplarını öncelikle sana verilen bağlam (context) içindeki bilgilere dayandır. "
        "Eğer bağlamda net bilgi yoksa, genel KVKK bilgisini kullanarak mantıklı ve açıklayıcı bir cevap üret. "
        "Yine de emin değilsen, 'Bağlamda bu soruya doğrudan yanıt bulunamadı.' de. "
        "Cevapların Türkçe, kısa ve öğretici olmalıdır.\n\n"
        "Bağlam (Context):\n{context}"
    )


    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # 6. RAG zincirini oluştur
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    st.success("✅ RAG zinciri başarıyla yüklendi!")
    return rag_chain


# ======================================================
# ===================  STREAMLIT UI  ===================
# ======================================================

st.set_page_config(page_title="KVKK Chatbot", page_icon="📄", layout="wide")
st.title("📄 KVKK RAG Chatbot")
st.caption("KVKK belgelerinizle konuşan yapay zekâ asistanı")

# --- Başlatma ---
try:
    api_key = get_api_key()
    configure_genai(api_key)
    rag_chain = load_rag_chain()
except Exception as e:
    st.error(f"Başlatma hatası: {e}")
    st.stop()

# --- Sohbet geçmişi ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Merhaba 👋 KVKK hakkındaki sorularınızı bana sorabilirsiniz."
    }]

# --- Mesaj geçmişini göster ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Kullanıcı girişi ---
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Asistan cevabı ---
    with st.chat_message("assistant"):
        with st.spinner("Belgeleri inceliyorum..."):
            try:
                response = rag_chain.invoke({"input": prompt})
                answer = (
                    response.get("answer") or
                    response.get("output_text") or
                    "Cevap üretilemedi."
                )
            except Exception as e:
                answer = f"⚠️ Hata oluştu: {e}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
