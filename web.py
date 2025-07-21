import streamlit as st
import pickle

# Fungsi untuk memuat model dan vectorizer
@st.cache_resource
def load_model_vectorizer():
    with open('tuned_count_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('tuned_naive_bayes_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model, vectorizer

# Load model dan vectorizer
model, vectorizer = load_model_vectorizer()

# Konfigurasi UI
st.set_page_config(page_title="Deteksi Spam Email", page_icon="📧")
st.title("📧 Deteksi Spam Email Otomatis")
st.markdown("""
Masukkan isi email di bawah untuk mengecek apakah termasuk **SPAM** atau bukan.  
🗒️ Harap gunakan **bahasa Inggris**, karena model dilatih dengan data berbahasa Inggris.
""")

# Input email
email_text = st.text_area("✉️ Masukkan isi email di sini:")

# Tombol deteksi
if st.button("🔍 Deteksi Sekarang"):
    if email_text.strip():
        try:
            # Preprocessing sederhana: lowercase
            cleaned_text = email_text.lower().strip()
            X_input = vectorizer.transform([cleaned_text])
            prediction = model.predict(X_input)[0]

            # Jika label hasil training berbentuk angka (0/1)
            if prediction == 1 or prediction == 'spam':
                st.error("🚫 Hasil: Email ini terdeteksi sebagai **SPAM**.")
            else:
                st.success("✅ Hasil: Email ini adalah **HAM** (bukan spam).")
        except Exception as e:
            st.error(f"❌ Gagal melakukan prediksi: {e}")
    else:
        st.warning("⚠️ Masukkan isi email terlebih dahulu.")

st.markdown("---")
st.markdown("🔐 Dibuat oleh **Devi Zahra Aulia** · Model: Multinomial Naive Bayes")
