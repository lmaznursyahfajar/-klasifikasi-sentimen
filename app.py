import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
import undetected_chromedriver as uc
from selenium.webdriver.support import expected_conditions as EC
from sklearn.feature_extraction.text import TfidfVectorizer
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
import os

# ==============================
# Layout & Sidebar
# ==============================
st.set_page_config(page_title="Sentimen Tokopedia", layout="wide")
st.sidebar.title("📌 Menu")
menu = st.sidebar.radio("Pilih Halaman:", ["📥 Klasifikasi Manual", "📁 Klasifikasi File CSV", "🔸 Scraping Tokopedia"])
st.title("💬 Aplikasi Klasifikasi Sentimen Tokopedia")

st.markdown("---")

# ==============================
# Load IndoBERT Model
# ==============================

stopwords_indonesia = [
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'untuk', 'dengan',
    'karena', 'pada', 'tidak', 'ada', 'saya', 'kami', 'kita', 'mereka',
    'juga', 'dalam', 'bisa', 'sudah', 'masih', 'jadi', 'lebih', 'kurang'
]

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()

    kata_map = {
        "mantap": ["mantap", "mantaaap", "mantapp", "mantaaaap", "mantappp", "mantaap", "mantaapp"],
        "bagus": ["baguus", "bagusss"],
        "cepat": ["cepeett", "cepattt", "cpet"],
        "lumayan": ["lumayannn"],
        "respon": ["responn", "responnn"],
    }

    for baku, variasi in kata_map.items():
        for kata in variasi:
            text = re.sub(rf'\b{kata}\b', baku, text)

    return text

@st.cache_resource
def load_pipeline():
    model = BertForSequenceClassification.from_pretrained("AznurOde21/indo-sentimen-tokopedia")
    tokenizer = BertTokenizer.from_pretrained("AznurOde21/indo-sentimen-tokopedia")
    return TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

pipeline = load_pipeline()

def decode_label(label):
    return "positif" if label.lower() == "label_1" else "negatif" if label.lower() == "label_0" else label

# ==============================
# Menu 1: Klasifikasi Manual
# ==============================
if menu == "📥 Klasifikasi Manual":
    st.subheader("✍️ Input Teks Manual")
    user_input = st.text_area("Tulis komentar atau ulasan:", height=150)

    if st.button("🔍 Klasifikasikan"):
        if user_input.strip() == "":
            st.warning("⚠️ Masukkan teks terlebih dahulu.")
        else:
            with st.spinner("Menganalisis..."):
                output = pipeline(user_input)[0]
                label_scores = {decode_label(item['label']): item['score'] for item in output}
                predicted_label = max(label_scores, key=label_scores.get)

            st.success("✅ Analisis selesai!")
            st.write(f"**Sentimen Prediksi:** `{predicted_label}`")
            st.write("**Probabilitas:**")
            for label, score in label_scores.items():
                st.write(f"- `{label}`: {score:.4f}")

# ==============================
# Menu 2: Klasifikasi CSV
# ==============================
elif menu == "📁 Klasifikasi File CSV":
    st.subheader("📂 Upload File CSV")
    uploaded_file = st.file_uploader("Unggah file CSV dengan kolom `comment`", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "comment" not in df.columns:
            st.error("❌ Kolom 'comment' tidak ditemukan.")
        else:
            st.write("📄 Contoh data:")
            st.dataframe(df.head())

            if st.button("🚀 Jalankan Klasifikasi"):
                with st.spinner("Menganalisis seluruh komentar..."):
                    results = pipeline(df['comment'].astype(str).tolist())
                    predicted_labels = [decode_label(max(r, key=lambda x: x['score'])['label']) for r in results]
                    df['predicted_sentiment'] = predicted_labels

                st.success("✅ Klasifikasi selesai!")
                st.dataframe(df[['comment', 'predicted_sentiment']])

                st.subheader("📊 Distribusi Sentimen")
                fig, ax = plt.subplots()
                sns.countplot(data=df, x='predicted_sentiment', palette='Set2', ax=ax)
                ax.set_title("Distribusi Sentimen")
                st.pyplot(fig)

                st.subheader("☁️ WordCloud")
                all_text = ' '.join(df['comment'].astype(str))
                normalized_text = normalize_text(all_text)
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(normalized_text)
                fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)

                st.subheader("🔠 10 Kata Terpenting dalam Klasifikasi")
                df['normalized_comment'] = df['comment'].astype(str).apply(normalize_text)
                vectorizer = TfidfVectorizer(stop_words=stopwords_indonesia, max_features=10)
                X = vectorizer.fit_transform(df['normalized_comment'])
                tfidf_dict = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1))
                sorted_items = sorted(tfidf_dict.items(), key=lambda item: item[1], reverse=True)
                words, scores = zip(*sorted_items)
                plt.figure(figsize=(10, 5))
                plt.barh(words, scores)
                plt.gca().invert_yaxis()
                st.pyplot(plt)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Unduh Hasil CSV", data=csv, file_name="hasil_klasifikasi.csv", mime='text/csv')

# ==============================
# Menu 3: Scraping Tokopedia (Pakai undetected-chromedriver)
# ==============================
elif menu == "🔸 Scraping Tokopedia":
    import undetected_chromedriver as uc
    from selenium.webdriver.common.by import By
    import time

    st.subheader("🔗 Scraping Review Tokopedia")
    url = st.text_input("Masukkan URL halaman produk Tokopedia:")

    if st.button("🔸 Mulai Scraping"):
        if not url.strip():
            st.warning("⚠️ Masukkan URL terlebih dahulu.")
        else:
            st.info("⏳ Sedang melakukan scraping...")

            try:
                options = uc.ChromeOptions()
                options.add_argument("--headless=new")  # mode tanpa GUI
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument("--window-size=1920x1080")

                driver = uc.Chrome(options=options)  # Chrome otomatis di-handle

                data = []
                driver.get(url)
                st.write("✅ Halaman dibuka, menunggu review muncul...")
                time.sleep(5)

                page = 1
                while True:
                    st.write(f"📄 Memproses halaman {page}...")

                    # Scroll agar review muncul
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(3)

                    review_elements = driver.find_elements(By.XPATH, '//div[@data-testid="lblItemUlasan"]')
                    st.write(f"🔍 Jumlah review ditemukan: {len(review_elements)}")

                    for review in review_elements:
                        text = review.text.strip()
                        if text:
                            data.append(text)

                    # Coba klik tombol berikutnya
                    try:
                        next_button = driver.find_element(By.XPATH, '//button[contains(text(), "Berikutnya")]')
                        if next_button.is_enabled():
                            next_button.click()
                            page += 1
                            time.sleep(3)
                        else:
                            break
                    except:
                        st.info("🔚 Tidak ada tombol 'Berikutnya' atau tidak bisa diklik.")
                        break

                driver.quit()

                # Tampilkan hasil
                if not data:
                    st.warning("❌ Tidak ada review ditemukan.")
                else:
                    st.success(f"✅ Ditemukan {len(data)} review.")
                    df = pd.DataFrame(data, columns=["comment"])
                    st.dataframe(df.head())

                    with st.spinner("🔍 Menganalisis sentimen..."):
                        results = pipeline(df['comment'].astype(str).tolist())
                        predicted_labels = [
                            decode_label(max(r, key=lambda x: x['score'])['label']) for r in results
                        ]
                        df['predicted_sentiment'] = predicted_labels

                    # Plot distribusi sentimen
                    st.subheader("📊 Distribusi Sentimen")
                    fig, ax = plt.subplots()
                    sns.countplot(data=df, x="predicted_sentiment", palette="Set2", ax=ax)
                    st.pyplot(fig)

                    # WordCloud
                    st.subheader("☁️ WordCloud")
                    all_text = " ".join(df['comment'].astype(str))
                    wc = WordCloud(width=800, height=300, background_color='white').generate(normalize_text(all_text))
                    fig_wc, ax_wc = plt.subplots(figsize=(10, 3))
                    ax_wc.imshow(wc, interpolation='bilinear')
                    ax_wc.axis("off")
                    st.pyplot(fig_wc)

                    # Unduh hasil
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Unduh File CSV",
                        data=csv,
                        file_name="hasil_scraping_tokopedia.csv",
                        mime='text/csv'
                    )

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat scraping: {e}")





