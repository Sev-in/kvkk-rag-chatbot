# 📄 KVKK RAG Chatbot

## 🎯 Projenin Amacı

Bu proje, **Kişisel Verilerin Korunması Kanunu (KVKK)** ile ilgili metinleri yapay zeka destekli olarak anlamak ve kullanıcılara doğru, bağlama dayalı yanıtlar sunmak amacıyla geliştirilmiştir.  
Proje, **Retrieval-Augmented Generation (RAG)** yaklaşımını kullanarak KVKK belgelerinden anlamlı bilgileri çıkarır ve sorulara bu belgelerdeki içeriğe dayanarak yanıt verir.  
Amaç; hukuk, veri gizliliği ve mevzuat konularında çalışan bireylerin, kanun metinleri arasında hızlı ve güvenilir şekilde bilgiye ulaşmalarını sağlamaktır.

---

## 📚 Veri Seti Hakkında

Veri seti, **Resmî Gazete** ve **Kişisel Verileri Koruma Kurumu (KVKK)** tarafından yayımlanan PDF formatındaki belgelerden oluşturulmuştur.  
Bu belgeler:

- Kanun metinleri
- Yönetmelikler
- Tebliğler
- Karar örnekleri
- Rehber dokümanları  
  gibi yasal içeriklerden oluşmaktadır.

Veri seti projenin `kvkk_veri_seti/` klasöründe tutulur ve işlendikten sonra vektör veritabanına (`chroma_db_kvkk/`) dönüştürülür.

---

## 🧠 Kullanılan Yöntemler

Proje, **Retrieval-Augmented Generation (RAG)** mimarisi üzerine kuruludur.  
Bu yapı aşağıdaki aşamalardan oluşur:

1. **PDF Temizleme ve Ön İşleme**

   - `unstructured` kütüphanesi ile PDF dosyalarından metin çıkarılır.
   - Gereksiz ifadeler (`Sayfa X / Y`, “T.C. Resmî Gazete” vb.) regex ile temizlenir.

2. **Vektörleştirme (Embedding)**

   - `Hugging Face` modeli olan `paraphrase-multilingual-MiniLM-L12-v2` ile Türkçe metinler yüksek doğrulukta gömülür.

3. **Veri Depolama (Vector Store)**

   - `Chroma` kullanılarak belgelerden elde edilen embedding’ler kalıcı şekilde saklanır.

4. **Sorgu (Retrieval)**

   - Kullanıcının sorduğu soru embedding’e dönüştürülür ve en alakalı metin parçaları Chroma’dan getirilir.

5. **Cevap Üretimi (Generation)**
   - `Google Gemini 2.0 Flash` modeli, sadece getirilen bağlamı kullanarak Türkçe bir yanıt oluşturur.

Tüm süreç LangChain altyapısı ile yönetilmiştir.

---

## 📊 Elde Edilen Sonuçlar

- KVKK belgelerindeki paragraflara dayalı olarak **doğru ve tutarlı yanıtlar** elde edilmiştir.
- Chatbot, mevzuat dışı sorulara karşı **“bağlamda bilgi bulunamadı”** şeklinde güvenli yanıtlar verebilmektedir.
- Türkçe metinlerde MiniLM embedding modeli, yüksek anlam koruma başarımı göstermiştir.
- Sistem, 1000+ sayfalık veri setinde **ortalama 1.2 saniye** yanıt süresiyle çalışmaktadır.

Proje sonucunda elde edilen chatbot, hukuk ve veri koruma alanlarında bilgiye erişimi hızlandıran **yerel olarak çalışan güvenli bir yapay zekâ aracı** haline gelmiştir.

---

## 🧩 Teknoloji Yığını

| Bileşen               | Kullanılan Teknoloji                                   |
| --------------------- | ------------------------------------------------------ |
| **LLM**               | Google Gemini (`gemini-2.0-flash-exp`)                 |
| **Embeddings**        | Hugging Face – `paraphrase-multilingual-MiniLM-L12-v2` |
| **RAG Framework**     | LangChain                                              |
| **Vektör Veritabanı** | Chroma                                                 |
| **Arayüz**            | Streamlit                                              |
| **PDF İşleme**        | Unstructured                                           |
| **Dil**               | Python 3.10+                                           |

---

## 📁 Proje Yapısı

````text
kvkk-rag-chatbot/
├── kvkk_veri_seti/             # PDF dataset (official KVKK documents)
├── chroma_db_kvkk/             # Persisted vector database (auto-created)
├── app.py                      # Streamlit app + RAG pipeline
├── veri_isleme.py              # PDF processing + embedding + storage
├── requirements.txt            # Dependencies
├── .env                        # GOOGLE_API_KEY (not committed)
└── README.md                   # Project documentation


## 🧩 Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/kvkk-rag-chatbot.git
cd kvkk-rag-chatbot
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Configure environment

Create a .env file in the project root:

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

### 4️⃣ Create the vector database

(Only required the first time or after data/model changes)

```bash
python veri_isleme.py
```

### 5️⃣ Start the app

```bash
streamlit run app.py
```


