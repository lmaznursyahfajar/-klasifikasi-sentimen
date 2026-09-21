import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import time
from datetime import datetime

# ==============================================================
# PAGE CONFIG
# ==============================================================
st.set_page_config(
    page_title="Analisis Sentimen Tokopedia",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================
# GLOBAL STYLING
# ==============================================================
CUSTOM_CSS = """
<style>
    /* ---------- General ---------- */
    .main {
        background-color: #F7F8FA;
    }
    #MainMenu, footer, header {visibility: hidden;}

    /* ---------- Typography ---------- */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Inter', sans-serif;
    }

    /* ---------- Hero header ---------- */
    .hero {
        background: linear-gradient(135deg, #00AA5B 0%, #049158 100%);
        padding: 2rem 2.2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(0, 170, 91, 0.25);
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .hero p {
        margin: 0.4rem 0 0 0;
        font-size: 1rem;
        opacity: 0.92;
    }

    /* ---------- Cards ---------- */
    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 10px rgba(16, 24, 40, 0.06);
        border: 1px solid #EEF0F2;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #111827;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #6B7280;
        margin-top: 0.15rem;
    }
    .section-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem 1.6rem;
        box-shadow: 0 2px 10px rgba(16, 24, 40, 0.06);
        border: 1px solid #EEF0F2;
        margin-bottom: 1.3rem;
    }

    /* ---------- Badges ---------- */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .badge-positif { background: #DCFCE7; color: #15803D; }
    .badge-negatif { background: #FEE2E2; color: #B91C1C; }
    .badge-netral  { background: #FEF3C7; color: #92400E; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
    }
    section[data-testid="stSidebar"] * {
        color: #E5E7EB !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
    }

    /* ---------- Buttons ---------- */
    div.stButton > button, .stDownloadButton > button {
        background: #00AA5B;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        transition: 0.2s;
    }
    div.stButton > button:hover, .stDownloadButton > button:hover {
        background: #049158;
        color: white;
    }

    /* ---------- Dataframe ---------- */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================================================
# CONSTANTS
# ==============================================================
SENTIMENT_COLORS = {"positif": "#00AA5B", "negatif": "#E53935", "netral": "#F5A623"}

STOPWORDS_ID = [
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'untuk', 'dengan',
    'karena', 'pada', 'tidak', 'ada', 'saya', 'kami', 'kita', 'mereka',
    'juga', 'dalam', 'bisa', 'sudah', 'masih', 'jadi', 'lebih', 'kurang',
    'akan', 'saat', 'atau', 'oleh', 'para', 'ya', 'nya', 'sih', 'aja',
]

KATA_MAP = {
    "mantap": ["mantap", "mantaaap", "mantapp", "mantaaaap", "mantappp", "mantaap", "mantaapp"],
    "bagus": ["baguus", "bagusss"],
    "cepat": ["cepeett", "cepattt", "cpet"],
    "lumayan": ["lumayannn"],
    "respon": ["responn", "responnn"],
}

# ==============================================================
# HELPERS
# ==============================================================
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    for baku, variasi in KATA_MAP.items():
        for kata in variasi:
            text = re.sub(rf'\b{kata}\b', baku, text)
    return text


def decode_label(label: str) -> str:
    label = label.lower()
    if label == "label_1":
        return "positif"
    if label == "label_0":
        return "negatif"
    return label


def best_score(result) -> dict:
    """Normalize a single pipeline result to one {'label', 'score'} dict.

    Depending on the transformers version / batch size, a `return_all_scores=True`
    (a.k.a. `top_k=None`) pipeline call can come back either as:
      - a list of {'label', 'score'} dicts (the expected shape), or
      - a single {'label', 'score'} dict (e.g. when the batch collapses to one item).
    This handles both so downstream code never has to guess.
    """
    if isinstance(result, dict):
        return result
    return max(result, key=lambda x: x["score"])


def all_scores_dict(result) -> dict:
    """Normalize a single pipeline result into a {label: score} dict for every class."""
    if isinstance(result, dict):
        return {decode_label(result["label"]): result["score"]}
    return {decode_label(item["label"]): item["score"] for item in result}


@st.cache_resource(show_spinner=False)
def load_pipeline():
    model = BertForSequenceClassification.from_pretrained("AznurOde21/indo-sentimen-tokopedia")
    tokenizer = BertTokenizer.from_pretrained("AznurOde21/indo-sentimen-tokopedia")
    return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)


def metric_card(col, value, label):
    col.markdown(
        f"""<div class="metric-card">
                <div class="value">{value}</div>
                <div class="label">{label}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def sentiment_badge(label: str) -> str:
    cls = {"positif": "badge-positif", "negatif": "badge-negatif"}.get(label, "badge-netral")
    return f'<span class="badge {cls}">{label.upper()}</span>'


def sentiment_pie(df, column="predicted_sentiment"):
    counts = df[column].value_counts().reset_index()
    counts.columns = ["Sentimen", "Jumlah"]
    fig = px.pie(
        counts, names="Sentimen", values="Jumlah", hole=0.55,
        color="Sentimen", color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_traces(textinfo="percent+label", pull=[0.03] * len(counts))
    fig.update_layout(
        showlegend=True, margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", y=-0.1),
    )
    return fig


def sentiment_bar(df, column="predicted_sentiment"):
    counts = df[column].value_counts().reset_index()
    counts.columns = ["Sentimen", "Jumlah"]
    fig = px.bar(
        counts, x="Sentimen", y="Jumlah", color="Sentimen",
        color_discrete_map=SENTIMENT_COLORS, text="Jumlah",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), yaxis_title="", xaxis_title="")
    return fig


def render_wordcloud(text_series, title="Word Cloud"):
    all_text = " ".join(text_series.astype(str))
    normalized = normalize_text(all_text)
    if not normalized.strip():
        st.info("Tidak cukup teks untuk membuat word cloud.")
        return
    wc = WordCloud(width=900, height=400, background_color="white", colormap="Greens").generate(normalized)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold")
    st.pyplot(fig, use_container_width=True)


def top_keywords_chart(df, text_col="comment"):
    df = df.copy()
    df["_normalized"] = df[text_col].astype(str).apply(normalize_text)
    non_empty = df["_normalized"].str.strip().astype(bool)
    if non_empty.sum() == 0:
        st.info("Tidak cukup teks untuk menghitung kata terpenting.")
        return
    vectorizer = TfidfVectorizer(stop_words=STOPWORDS_ID, max_features=10)
    X = vectorizer.fit_transform(df.loc[non_empty, "_normalized"])
    tfidf_dict = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1))
    sorted_items = sorted(tfidf_dict.items(), key=lambda item: item[1], reverse=True)
    words, scores = zip(*sorted_items)
    fig = px.bar(
        x=list(scores), y=list(words), orientation="h",
        labels={"x": "Skor TF-IDF", "y": ""}, color=list(scores),
        color_continuous_scale="Greens",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False,
                       coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)


def run_batch_classification(pipe, comments: pd.Series) -> tuple[list, list]:
    """Classify comments in batches with a live progress bar."""
    texts = comments.astype(str).tolist()
    predicted_labels, confidences = [], []
    batch_size = 16
    progress = st.progress(0, text="Memulai analisis...")
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        results = pipe(batch)
        # `results` should be a list with one entry (itself a list of per-label
        # score dicts) per input text. Guard against shape quirks when the
        # batch has just one item: the pipeline can collapse the outer list
        # and hand back either a single dict, or the flat list of per-label
        # dicts for that one item instead of a list-of-lists.
        if isinstance(results, dict):
            results = [results]
        elif len(batch) == 1 and isinstance(results, list) and results and isinstance(results[0], dict):
            results = [results]
        for r in results:
            best = best_score(r)
            predicted_labels.append(decode_label(best["label"]))
            confidences.append(best["score"])
        done = min(i + batch_size, total)
        progress.progress(done / total, text=f"Menganalisis komentar... {done}/{total}")
    progress.empty()
    return predicted_labels, confidences


def show_results_dashboard(df: pd.DataFrame, key_prefix: str):
    """Shared dashboard: metrics, charts, wordcloud, keywords, download."""
    total = len(df)
    n_pos = (df["predicted_sentiment"] == "positif").sum()
    n_neg = (df["predicted_sentiment"] == "negatif").sum()
    pos_pct = f"{n_pos / total * 100:.1f}%" if total else "0%"

    st.markdown("#### 📊 Ringkasan")
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, total, "Total Komentar")
    metric_card(c2, n_pos, "Sentimen Positif")
    metric_card(c3, n_neg, "Sentimen Negatif")
    metric_card(c4, pos_pct, "Rasio Positif")

    st.write("")
    col_pie, col_bar = st.columns(2)
    with col_pie:
        st.markdown("**Distribusi Sentimen**")
        st.plotly_chart(sentiment_pie(df), use_container_width=True, key=f"{key_prefix}_pie")
    with col_bar:
        st.markdown("**Jumlah per Kategori**")
        st.plotly_chart(sentiment_bar(df), use_container_width=True, key=f"{key_prefix}_bar")

    st.markdown("---")
    st.markdown("#### ☁️ Word Cloud & Kata Kunci")
    wc_col, kw_col = st.columns([1.2, 1])
    with wc_col:
        render_wordcloud(df["comment"])
    with kw_col:
        top_keywords_chart(df)

    st.markdown("---")
    st.markdown("#### 📄 Data Lengkap")
    st.dataframe(df, use_container_width=True, height=320)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Unduh Hasil (CSV)", data=csv,
        file_name=f"hasil_klasifikasi_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv", key=f"{key_prefix}_download",
    )


# ==============================================================
# SIDEBAR
# ==============================================================
with st.sidebar:
    st.markdown("## 💬 Menu Navigasi")
    menu = st.radio(
        "",
        ["📥 Klasifikasi Manual", "📁 Klasifikasi File CSV"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ℹ️ Tentang Aplikasi")
    st.markdown(
        "Aplikasi ini mengklasifikasikan sentimen ulasan pelanggan "
        "Tokopedia menggunakan model **IndoBERT** yang telah dilatih "
        "khusus untuk teks berbahasa Indonesia."
    )
    st.markdown("---")
    st.caption("Model: `AznurOde21/indo-sentimen-tokopedia`")
    st.caption(f"Sesi dibuka: {datetime.now().strftime('%d %b %Y, %H:%M')}")

# ==============================================================
# HERO HEADER
# ==============================================================
st.markdown(
    """
    <div class="hero">
        <h1>💬 Analisis Sentimen Ulasan Tokopedia</h1>
        <p>Klasifikasikan ulasan pelanggan secara instan menggunakan model IndoBERT — lengkap dengan
        visualisasi, word cloud, dan ekspor hasil.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================
# LOAD MODEL (once)
# ==============================================================
with st.spinner("🔄 Memuat model IndoBERT, mohon tunggu..."):
    sentiment_pipeline = load_pipeline()

# ==============================================================
# MENU 1 — KLASIFIKASI MANUAL
# ==============================================================
if menu == "📥 Klasifikasi Manual":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### ✍️ Input Teks Manual")
    st.write("Masukkan satu komentar atau ulasan pelanggan untuk dianalisis sentimennya.")

    user_input = st.text_area(
        "Komentar:", height=140,
        placeholder="Contoh: Barangnya bagus banget, pengiriman cepat, seller ramah!",
        label_visibility="collapsed",
    )

    col_btn, _ = st.columns([1, 4])
    analyze_clicked = col_btn.button("🔍 Klasifikasikan", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if analyze_clicked:
        if not user_input.strip():
            st.warning("⚠️ Masukkan teks terlebih dahulu.")
        else:
            with st.spinner("Menganalisis..."):
                output = sentiment_pipeline(user_input)
                # Some transformers versions return a single dict, others a
                # list-of-one containing the list of per-label scores.
                if isinstance(output, list) and len(output) == 1 and isinstance(output[0], list):
                    output = output[0]
                label_scores = all_scores_dict(output) if isinstance(output, dict) else {
                    decode_label(item["label"]): item["score"] for item in output
                }
                predicted_label = max(label_scores, key=label_scores.get)
                confidence = label_scores[predicted_label]

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.markdown("**Hasil Prediksi:**")
                st.markdown(sentiment_badge(predicted_label), unsafe_allow_html=True)
                st.metric("Tingkat Keyakinan", f"{confidence * 100:.1f}%")
            with res_col2:
                st.markdown("**Distribusi Probabilitas:**")
                for label, score in sorted(label_scores.items(), key=lambda x: -x[1]):
                    st.write(f"`{label}`")
                    st.progress(float(score), text=f"{score * 100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================
# MENU 2 — KLASIFIKASI FILE CSV
# ==============================================================
elif menu == "📁 Klasifikasi File CSV":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 📂 Unggah File CSV")
    st.write("File harus memiliki kolom bernama **`comment`** yang berisi teks ulasan.")

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")
            df = None

        if df is not None:
            if "comment" not in df.columns:
                st.error("❌ Kolom `comment` tidak ditemukan pada file yang diunggah.")
            else:
                df = df.dropna(subset=["comment"]).reset_index(drop=True)
                st.success(f"✅ File berhasil dimuat — {len(df)} baris ditemukan.")
                st.dataframe(df.head(5), use_container_width=True)

                run_clicked = st.button("🚀 Jalankan Klasifikasi", use_container_width=False)
                st.markdown('</div>', unsafe_allow_html=True)

                if run_clicked:
                    if len(df) == 0:
                        st.warning("⚠️ Tidak ada data untuk dianalisis.")
                    else:
                        start = time.time()
                        predicted_labels, confidences = run_batch_classification(sentiment_pipeline, df["comment"])
                        df["predicted_sentiment"] = predicted_labels
                        df["confidence"] = [f"{c * 100:.1f}%" for c in confidences]
                        elapsed = time.time() - start

                        st.success(f"✅ Klasifikasi selesai dalam {elapsed:.1f} detik.")
                        show_results_dashboard(df, key_prefix="csv")
        else:
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("👆 Unggah file CSV untuk memulai analisis.")
        st.markdown('</div>', unsafe_allow_html=True)
