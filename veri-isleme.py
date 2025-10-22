import os
import glob
import re
import uuid
from tqdm import tqdm
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
import chromadb
import google.generativeai as genai

# ---------- ORTAM AYARLARI ----------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY bulunamadı. Lütfen .env dosyana ekle: GOOGLE_API_KEY=...")

genai.configure(api_key=API_KEY)

CHROMA_DIR = "chroma_db_kvkk"
COLLECTION_NAME = "kvkk_docs"

# ---------- TEMİZLİK AYARLARI ----------
TEMIZLENECEK_KALIPLAR = [
    r"Sayfa \d+ / \d+",
    r"T\.C\. Resmi Gazete",
]

def metni_regex_ile_temizle(metin: str) -> str:
    temiz = metin
    for kalip in TEMIZLENECEK_KALIPLAR:
        temiz = re.sub(kalip, '', temiz)
    temiz = re.sub(r'\s{2,}', ' ', temiz)
    temiz = re.sub(r'\n{3,}', '\n\n', temiz)
    return temiz.strip()

# ---------- GEMINI EMBEDDING FONKSİYONU ----------
def get_embedding(text: str, model: str = "models/embedding-001"):
    """
    Gemini'nin embedding API'sini kullanarak bir metin için embedding döndürür.
    """
    try:
        result = genai.embed_content(
            model=model,
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]
    except Exception as e:
        print(f"⚠️ Embedding hatası: {e}")
        return None

# ---------- CHROMA BAĞLANTISI ----------
def init_chroma():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if COLLECTION_NAME not in [c.name for c in client.list_collections()]:
        col = client.create_collection(name=COLLECTION_NAME)
    else:
        col = client.get_collection(COLLECTION_NAME)
    return col

# ---------- PDF TEMİZLEME + VEKÖR KAYIT ----------
def pdfleri_isle_ve_kaydet(ana_klasor: str, strateji: str = "hi_res"):
    print(f"'{ana_klasor}' taranıyor... Strateji: {strateji}")

    pdf_yollar = glob.glob(os.path.join(ana_klasor, "**", "*.pdf"), recursive=True)
    if not pdf_yollar:
        print("❌ Hiç PDF bulunamadı.")
        return

    print(f"📄 Toplam {len(pdf_yollar)} PDF bulundu. İşlem başlıyor...")
    col = init_chroma()

    for pdf_path in tqdm(pdf_yollar, desc="PDF işleniyor"):
        try:
            elements = partition_pdf(filename=pdf_path, strategy=strateji, languages=["tur"])
            ham_metin = ""
            for el in elements:
                t = str(type(el)).lower()
                if any(k in t for k in ["title", "narrativetext", "listitem"]):
                    ham_metin += el.text + "\n\n"

            temiz_metin = metni_regex_ile_temizle(ham_metin)

            # Fazla uzun metinleri küçük parçalara bölelim
            kelimeler = temiz_metin.split()
            chunk_size = 500
            chunks = [" ".join(kelimeler[i:i+chunk_size]) for i in range(0, len(kelimeler), chunk_size)]

            for ch in chunks:
                emb = get_embedding(ch)
                if emb:
                    col.add(
                        ids=[str(uuid.uuid4())],
                        embeddings=[emb],
                        metadatas=[{"source": pdf_path}],
                        documents=[ch],
                    )

        except Exception as e:
            print(f"⚠️ Hata: {pdf_path} işlenemedi: {e}")
            continue

    print(f"✅ Tüm PDF'ler işlendi ve '{CHROMA_DIR}' altına kaydedildi.")

# ---------- ANA ÇALIŞTIRMA ----------
if __name__ == "__main__":
    klasor = "kvkk_veri_Seti"  # PDF'lerin bulunduğu klasör
    pdfleri_isle_ve_kaydet(klasor, strateji="hi_res")
