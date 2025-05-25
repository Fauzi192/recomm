import streamlit as st
import pandas as pd
import pickle
import random

# ---------- Konfigurasi Halaman ----------
st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎌",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- CSS untuk Styling ----------
st.markdown("""
    <style>
        .big-title {
            font-size: 48px;
            font-weight: bold;
            color: #F63366;
            text-align: center;
        }
        .anime-card {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        }
        .anime-title {
            font-size: 20px;
            font-weight: bold;
            color: #2C3E50;
        }
        .genre-text {
            font-size: 16px;
            color: #7F8C8D;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Fungsi Rekomendasi ----------
def get_knn_recommendations(anime_title, knn_model, df, n=5):
    index = df[df['name'] == anime_title].index[0]
    distances, indices = knn_model.kneighbors([df.iloc[index, 2:]], n_neighbors=n+1)
    rekomendasi = df.iloc[indices[0][1:]][['name', 'genre']]
    return rekomendasi

# ---------- Load Model dan Dataset ----------
with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

anime = pd.read_csv('anime.csv')

# ---------- Pilih Judul Acak ----------
judul_input = random.choice(anime['name'].tolist())
genre_input = anime[anime['name'] == judul_input]['genre'].values[0]

# ---------- UI ----------
st.markdown('<div class="big-title">🎌 Anime Recommendation Engine</div>', unsafe_allow_html=True)
st.markdown("")

# ---------- Anime yang Dipilih ----------
st.markdown('<div class="anime-card">', unsafe_allow_html=True)
st.markdown(f"<div class='anime-title'>🎯 Anime Terpilih: {judul_input}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='genre-text'>📚 Genre: {genre_input}</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tampilkan Rekomendasi ----------
st.subheader("🔁 Rekomendasi Anime Serupa:")
rekomendasi_df = get_knn_recommendations(judul_input, knn_model, anime, n=5)

for idx, row in rekomendasi_df.iterrows():
    st.markdown('<div class="anime-card">', unsafe_allow_html=True)
    st.markdown(f"<div class='anime-title'>🎥 {row['name']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='genre-text'>📌 Genre: {row['genre']}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
