# 💬 Aplikasi Klasifikasi Sentimen Tokopedia

Aplikasi web berbasis **Streamlit** untuk menganalisis sentimen ulasan atau komentar Tokopedia secara otomatis. Aplikasi ini memanfaatkan model Transformer **IndoBERT** (`AznurOde21/indo-sentimen-tokopedia`) dari Hugging Face untuk mengklasifikasikan teks ke dalam kategori **Positif** atau **Negatif**.

---

## ✨ Fitur Utama

- **📥 Klasifikasi Manual**: Prediksi sentimen teks tunggal secara langsung beserta tingkat probabilitas probabilitasnya.
- **📁 Klasifikasi File CSV**: Proses klasifikasi secara massal (*batch processing*) dari unggahan file CSV, dilengkapi visualisasi **Distribusi Sentimen**, **WordCloud**, dan **10 Kata Terpenting (TF-IDF)**.
- **🔸 Scraping Tokopedia**: Pengambilan data ulasan langsung dari URL produk Tokopedia menggunakan web scraping yang terintegrasi dengan analisis sentimen instan.

---

## 📦 Prasyarat & Instalasi

### 1. Requirements

Buat file bernama `requirements.txt` dan masukkan pustaka berikut:

```text
streamlit
pandas
matplotlib
seaborn
wordcloud
transformers
torch
requests
beautifulsoup4
scikit-learn
undetected-chromedriver
webdriver-manager
