import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, TextClassificationPipeline
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# =======================
# Load Model & Tokenizer
# =======================
@st.cache_resource
def load_pipeline():
    model = BertForSequenceClassification.from_pretrained("AznurOde21/indo-sentimen-tokopedia")
    tokenizer = BertTokenizer.from_pretrained("AznurOde21/indo-sentimen-tokopedia")
    return TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

pipeline = load_pipeline()

def decode_label(label):
    if label.lower() == "label_1":
        return "positif"
    elif label.lower() == "label_0":
        return "negatif"
    return label

# =======================
# Sidebar Menu
# =======================
st.sidebar.title("📌 Menu")
menu = st.sidebar.radio("Pilih Halaman:", ["Klasifikasi Manual", "Klasifikasi File CSV"])

st.title("💬 Aplikasi Klasifikasi Sentimen IndoBERT")

# =======================
# Menu 1: Klasifikasi Manual
# =======================
if menu == "Klasifikasi Manual":
    st.header("✍️ Input Teks Manual")
    user_input = st.text_area("Tulis komentar atau ulasan:", height=150)

    if st.button("🔍 Klasifikasikan"):
        if user_input.strip() == "":
            st.warning("Masukkan teks terlebih dahulu.")
        else:
            with st.spinner("Menganalisis..."):
                output = pipeline(user_input)[0]
                label_scores = {decode_label(item['label']): item['score'] for item in output}
                predicted_label = max(label_scores, key=label_scores.get)

                st.subheader("📊 Hasil Klasifikasi")
                st.write(f"**Sentimen Prediksi:** `{predicted_label}`")
                st.write("**Probabilitas:**")
                for label, score in label_scores.items():
                    st.write(f"- `{label}`: {score:.4f}")

# =======================
# Menu 2: Klasifikasi File CSV
# =======================
elif menu == "Klasifikasi File CSV":
    st.header("📂 Upload File CSV")

    uploaded_file = st.file_uploader("Unggah file CSV dengan kolom `comment`", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "comment" not in df.columns:
            st.error("❌ Kolom 'comment' tidak ditemukan.")
        else:
            st.write("Contoh data:")
            st.dataframe(df.head())

            if st.button("🚀 Jalankan Klasifikasi"):
                with st.spinner("Menganalisis seluruh komentar..."):
                    results = pipeline(df['comment'].astype(str).tolist())
                    predicted_labels = [decode_label(max(r, key=lambda x: x['score'])['label']) for r in results]
                    df['predicted_sentiment'] = predicted_labels

                st.success("✅ Klasifikasi selesai!")
                st.subheader("📋 Hasil Klasifikasi")
                st.dataframe(df[['comment', 'predicted_sentiment']])

                # ====== Visualisasi: Bar Chart ======
                st.subheader("📊 Distribusi Sentimen")
                fig, ax = plt.subplots()
                sns.countplot(data=df, x="predicted_sentiment", palette="Set2", ax=ax)
                ax.set_xlabel("Sentimen")
                ax.set_ylabel("Jumlah")
                ax.set_title("Distribusi Sentimen")
                st.pyplot(fig)

                # ====== Visualisasi: WordCloud ======
                st.subheader("☁️ WordCloud per Sentimen")
                for sentimen in df['predicted_sentiment'].unique():
                    st.markdown(f"**Sentimen: {sentimen.capitalize()}**")
                    text_data = " ".join(df[df['predicted_sentiment'] == sentimen]['comment'].dropna().astype(str).tolist())
                    if text_data.strip():
                        wordcloud = WordCloud(width=800, height=300, background_color='white', collocations=False).generate(text_data)
                        fig_wc, ax_wc = plt.subplots(figsize=(10, 3))
                        ax_wc.imshow(wordcloud, interpolation='bilinear')
                        ax_wc.axis("off")
                        st.pyplot(fig_wc)
                    else:
                        st.write("_Tidak ada kata untuk ditampilkan._")

                # ====== Download Hasil ======
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Unduh Hasil", data=csv, file_name="hasil_sentimen.csv", mime='text/csv')