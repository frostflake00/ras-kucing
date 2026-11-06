"""
Aplikasi Web Klasifikasi Ras Kucing
Birman | Ragdoll | Siamese
Menggunakan MobileNetV2 + SVM (Oxford-IIIT Pet Dataset)
"""

import streamlit as st
import numpy as np
import joblib
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import os

# ─── Konfigurasi Halaman ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Ras Kucing",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Path ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "..", "saved_models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

MODEL_SVM_PATH = os.path.join(MODEL_DIR, "model_svm_best.joblib")
LABEL_ENC_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
LOGO_PATH      = os.path.join(ASSETS_DIR, "logo.png")

# ─── Informasi Ras Kucing ─────────────────────────────────────────────────────
BREED_INFO = {
    "Birman": {
        "gambar": os.path.join(ASSETS_DIR, "birman.png"),
        "ciri_khas": "Bulu semi-panjang, mata biru safir, dan memiliki sarung tangan putih (gloves) murni pada keempat telapak kakinya.",
        "deskripsi": (
            "Kucing Birman (Kucing Suci Burma) memiliki ciri khas visual berupa warna point pada telinga, wajah, dan ekor, "
            "tetapi uniknya selalu memiliki 'sarung tangan' putih bersih di keempat telapak kakinya. "
            "Karakternya sangat tenang, setia, bersuara lembut, dan bersahabat."
        ),
    },
    "Ragdoll": {
        "gambar": os.path.join(ASSETS_DIR, "ragdoll.png"),
        "ciri_khas": "Ukuran tubuh besar dan berotot, bulu lebat lembut seperti kelinci, mata biru oval, serta tubuh lemas saat digendong.",
        "deskripsi": (
            "Ragdoll adalah ras kucing berukuran besar dengan bulu semi-panjang yang sangat lembut. "
            "Diberi nama 'Ragdoll' karena tubuhnya akan melemas rileks bagaikan boneka kain saat dipeluk. "
            "Sangat jinak, suka mengikuti pemiliknya ke mana saja, dan ideal untuk keluarga."
        ),
    },
    "Siamese": {
        "gambar": os.path.join(ASSETS_DIR, "siam.png"),
        "ciri_khas": "Tubuh ramping oriental (athletic/slender), bulu pendek halus, wajah segitiga, dan mata biru tajam berbentuk almond.",
        "deskripsi": (
            "Siamese (Siam) berasal dari Thailand dan merupakan ras tertua bertipe tubuh oriental. "
            "Berbeda dari Birman dan Ragdoll yang berbulu panjang/sedang, Siamese memiliki bulu pendek yang rapat "
            "dengan kontras warna gelap di ujung tubuh (point). Kucing ini sangat vokal, cerdas, dan aktif."
        ),
    },
}

# ─── Custom CSS (Clean White Background + Toska Cyan Accents) ────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Background Utama Bersih Putih */
    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1150px !important;
    }

    #MainMenu, footer, header {
        visibility: hidden !important;
        height: 0px !important;
    }

    /* Navbar Bersih */
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 1.2rem;
        margin-bottom: 1.8rem;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 1rem;
    }

    /* Kartu Putih Minimalis */
    .clean-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.2rem;
    }

    .clean-card-accent {
        background: #F8FAFC;
        border-left: 5px solid #00BCD4;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
        box-shadow: 0 2px 8px rgba(0, 188, 212, 0.06);
    }

    .clean-card-cyan {
        background: linear-gradient(135deg, #00BCD4 0%, #0097A7 100%);
        color: white !important;
        border-radius: 18px;
        padding: 1.6rem;
        box-shadow: 0 8px 20px rgba(0, 188, 212, 0.25);
    }

    /* Headings */
    .title-cyan {
        color: #00ACC1;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.3px;
        margin-bottom: 0.4rem;
    }

    .section-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }

    /* Tombol Kustom */
    div.stButton > button {
        border-radius: 30px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px !important;
        padding: 0.55rem 1.8rem !important;
        transition: all 0.2s ease-in-out !important;
    }

    /* Tombol Nav Aktif & Nonaktif */
    div.stButton > button[kind="primary"] {
        background-color: #00BCD4 !important;
        color: #FFFFFF !important;
        border: 1px solid #00BCD4 !important;
        box-shadow: 0 4px 12px rgba(0, 188, 212, 0.3) !important;
    }

    div.stButton > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #64748B !important;
        border: 1px solid #E2E8F0 !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        color: #00BCD4 !important;
        border-color: #00BCD4 !important;
        background-color: #F0FDFA !important;
    }

    /* Badge & Metrik Minimalis */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }
    .badge-cyan {
        background: #E0F7FA;
        color: #00838F;
    }
    .badge-gray {
        background: #F1F5F9;
        color: #475569;
    }

    /* Custom Progress Bar Styling */
    .stProgress > div > div > div > div {
        background-color: #00BCD4 !important;
        border-radius: 10px !important;
    }

    /* Tab Kustom */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F1F5F9;
        padding: 5px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        color: #64748B;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #00BCD4 !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }

    /* Kotak Hasil Prediksi */
    .pred-box {
        background: linear-gradient(135deg, #00BCD4 0%, #0097A7 100%);
        border-radius: 16px;
        padding: 1.4rem;
        color: white;
        text-align: center;
        box-shadow: 0 6px 18px rgba(0, 188, 212, 0.25);
    }
    .pred-title {
        font-size: 0.85rem;
        letter-spacing: 1.5px;
        font-weight: 700;
        text-transform: uppercase;
        opacity: 0.9;
        margin-bottom: 0.2rem;
    }
    .pred-label {
        font-size: 2rem;
        font-weight: 900;
        letter-spacing: 1px;
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ─── Load Model (Cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load MobileNetV2 feature extractor + SVM classifier."""
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    svm_pipeline = joblib.load(MODEL_SVM_PATH)
    label_encoder = joblib.load(LABEL_ENC_PATH)

    return base_model, svm_pipeline, label_encoder

# ─── Fungsi Prediksi ──────────────────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    arr = img_to_array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict(pil_image: Image.Image):
    base_model, svm_pipeline, label_encoder = load_models()

    img_array = preprocess_image(pil_image)
    features = base_model.predict(img_array, verbose=0)

    pred_idx = svm_pipeline.predict(features)[0]
    predicted_label = label_encoder.inverse_transform([pred_idx])[0]

    # Probabilitas via softmax pada decision function SVM
    decision = svm_pipeline.decision_function(features)[0]
    exp_d = np.exp(decision - np.max(decision))
    proba_arr = exp_d / exp_d.sum()

    classes = label_encoder.inverse_transform(np.arange(len(proba_arr)))
    proba = {cls: float(prob * 100) for cls, prob in zip(classes, proba_arr)}

    return predicted_label, proba

# ─── Render Logo ──────────────────────────────────────────────────────────────
def render_logo(max_width=520, center=True):
    if os.path.exists(LOGO_PATH):
        if center:
            col1, col2, col3 = st.columns([1, 2.2, 1])
            with col2:
                st.image(LOGO_PATH, use_container_width=True)
        else:
            st.image(LOGO_PATH, width=max_width)

# ─── Navbar ───────────────────────────────────────────────────────────────────
def navbar():
    current = st.session_state.get("page", "Home")

    cols = st.columns([1, 1, 1, 1, 1])
    with cols[1]:
        if st.button("HOME", key="nav_home",
                     type="primary" if current == "Home" else "secondary",
                     use_container_width=True):
            st.session_state["page"] = "Home"
            st.rerun()
    with cols[2]:
        if st.button("STATISTIK", key="nav_stat",
                     type="primary" if current == "Statistik" else "secondary",
                     use_container_width=True):
            st.session_state["page"] = "Statistik"
            st.rerun()
    with cols[3]:
        if st.button("ABOUT", key="nav_about",
                     type="primary" if current == "About" else "secondary",
                     use_container_width=True):
            st.session_state["page"] = "About"
            st.rerun()

# ─── Halaman 1: HOME ──────────────────────────────────────────────────────────
def page_home():
    col_left, col_right = st.columns([1.1, 1.0], gap="large")

    with col_left:
        # Tampilkan logo resmi yang diupload
        render_logo(center=False)

        # Tab Info Ras Bergantian
        st.markdown('<div class="section-title" style="margin-top:0.8rem;">Mengenal 3 Ras Serupa</div>', unsafe_allow_html=True)
        tabs = st.tabs(["Birman", "Ragdoll", "Siamese"])

        for i, breed in enumerate(["Birman", "Ragdoll", "Siamese"]):
            with tabs[i]:
                info = BREED_INFO[breed]
                st.markdown(
                    f'<div class="clean-card-accent">'
                    f'<span class="badge badge-cyan">{breed}</span>'
                    f'<p style="margin: 0.6rem 0 0.4rem 0; font-size: 0.93rem; color: #334155; line-height: 1.6;">'
                    f'{info["deskripsi"]}'
                    f'</p>'
                    f'<p style="margin: 0; font-size: 0.85rem; color: #0284C7; font-weight: 600;">'
                    f'✨ Ciri Utama: {info["ciri_khas"]}'
                    f'</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with col_right:
        st.markdown(
            '<div class="clean-card" style="border-top: 4px solid #00BCD4;">'
            '<div class="section-title">Cari Tahu Kucingmu Jenis Apa</div>'
            '<div class="section-subtitle">Pilih metode pengambilan gambar di bawah ini:</div>',
            unsafe_allow_html=True,
        )

        tab_browse,tab_cam = st.tabs(["Browse File","Camera"])

        with tab_browse:
            st.caption("Unggah foto kucing format JPG, PNG, atau WEBP:")
            uploaded = st.file_uploader(
                "Upload Gambar",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
            )
            if uploaded is not None:
                pil_img = Image.open(uploaded)
                _handle_input_image(pil_img, key_suffix="browse")

        with tab_cam:
            st.caption("Arahkan kamera ke wajah atau tubuh kucing:")
            cam_image = st.camera_input("Ambil Foto", label_visibility="collapsed")
            if cam_image is not None:
                pil_img = Image.open(cam_image)
                _handle_input_image(pil_img, key_suffix="cam")

        st.markdown('</div>', unsafe_allow_html=True)


def _handle_input_image(pil_img: Image.Image, key_suffix=""):
    st.image(pil_img, caption="Pratinjau Foto", use_container_width=True)

    if st.button("🐾  KLASIFIKASIKAN SEKARANG", key=f"btn_classify_{key_suffix}", type="primary", use_container_width=True):
        with st.spinner("Sedang mengekstraksi fitur dan mengklasifikasikan..."):
            label, proba = predict(pil_img)

        st.session_state["last_result"] = {
            "image": pil_img,
            "label": label,
            "proba": proba,
        }
        st.session_state["page"] = "Statistik"
        st.rerun()

# ─── Halaman 2: STATISTIK ─────────────────────────────────────────────────────
def page_statistik():
    render_logo(max_width=420, center=True)

    result = st.session_state.get("last_result", None)
    if result is None:
        st.markdown(
            '<div class="clean-card" style="text-align: center; padding: 2.5rem 1rem;">'
            '<h4>Belum Ada Gambar yang Diklasifikasikan</h4>'
            '<p style="color: #64748B;">Silakan buka halaman Home lalu ambil foto atau unggah gambar kucing terlebih dahulu.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns([1.5, 1, 1.5])
        with c2:
            if st.button("⬅️  Kembali ke Home", type="primary", use_container_width=True):
                st.session_state["page"] = "Home"
                st.rerun()
        return

    label = result["label"]
    proba = result["proba"]
    pil_img = result["image"]

    st.markdown('<div style="height: 1.2rem;"></div>', unsafe_allow_html=True)
    col_bar, col_result = st.columns([1.3, 1.0], gap="large")

    with col_bar:
        st.markdown(
            '<div class="clean-card">'
            '<div class="section-title">Persentase Kemiripan</div>'
            '<div class="section-subtitle">Tingkat keyakinan model SVM terhadap masing-masing ras:</div>',
            unsafe_allow_html=True,
        )

        sorted_proba = sorted(proba.items(), key=lambda x: x[1], reverse=True)

        for breed, pct in sorted_proba:
            breed_img = BREED_INFO[breed]["gambar"]
            is_top = (breed == label)

            c_ico, c_bar = st.columns([0.16, 0.84])
            with c_ico:
                if os.path.exists(breed_img):
                    st.image(breed_img, width=65)
            with c_bar:
                highlight_style = "color: #00838F; font-weight: 800;" if is_top else "color: #1E293B; font-weight: 600;"
                st.markdown(
                    f'<div style="display:flex; justify-content:space-between; margin-bottom:4px;">'
                    f'<span style="{highlight_style} font-size:1rem;">{breed} {"🏆" if is_top else ""}</span>'
                    f'<span style="font-weight:700; color:#00BCD4;">{pct:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.progress(int(round(pct)))
                st.markdown(f'<div style="font-size:0.8rem; color:#64748B; margin-top:-4px; margin-bottom:12px;">{BREED_INFO[breed]["ciri_khas"]}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown(
            f'<div class="pred-box">'
            f'<div class="pred-title">Hasil Klasifikasi</div>'
            f'<div class="pred-label">{label.upper()}</div>'
            f'<div style="font-size:0.9rem; margin-top:4px; opacity:0.95;">Keyakinan: <b>{proba[label]:.1f}%</b></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height: 0.8rem;"></div>', unsafe_allow_html=True)
        st.image(pil_img, caption="Citra Input Pengujian", use_container_width=True)

    # Card Penjelasan Ras
    st.markdown(
        f'<div class="clean-card-accent" style="margin-top:1.5rem;">'
        f'<div class="section-title" style="font-size:1.15rem; color:#00838F;">Tentang Ras {label}</div>'
        f'<p style="color:#334155; line-height:1.6; margin-top:0.4rem;">{BREED_INFO[label]["deskripsi"]}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1.5, 1, 1.5])
    with c2:
        if st.button("🔄  Uji Gambar Lain", type="secondary", use_container_width=True):
            st.session_state["last_result"] = None
            st.session_state["page"] = "Home"
            st.rerun()

# ─── Halaman 3: ABOUT (Minimalist & Simple Style) ─────────────────────────────
def page_about():
    render_logo(max_width=450, center=True)

    st.markdown(
        '<div style="text-align: center; max-width: 720px; margin: 0.5rem auto 1.8rem auto;">'
        '<p style="font-size: 1.05rem; color: #475569; line-height: 1.6;">'
        'Sistem klasifikasi cerdas untuk membedakan tiga ras kucing dengan pola warna serupa '
        '(<b>Birman</b>, <b>Ragdoll</b>, dan <b>Siamese</b>) '
        'menggunakan kombinasi Deep Learning sebagai ekstraktor fitur dan Support Vector Machine (SVM) sebagai pengklasifikasi.'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # 3 Metrik Utama Berjejer Bersih
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            '<div class="clean-card" style="text-align:center; padding:1.2rem;">'
            '<span class="badge badge-cyan">Model SVM</span>'
            '<h2 style="color:#00BCD4; margin:0.5rem 0 0.2rem 0; font-weight:800;">87.5%</h2>'
            '<p style="color:#64748B; font-size:0.85rem; margin:0;">Akurasi Uji (RBF Kernel)</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            '<div class="clean-card" style="text-align:center; padding:1.2rem;">'
            '<span class="badge badge-cyan">Feature Extractor</span>'
            '<h2 style="color:#00BCD4; margin:0.5rem 0 0.2rem 0; font-weight:800;">1.280</h2>'
            '<p style="color:#64748B; font-size:0.85rem; margin:0;">Dimensi Fitur (MobileNetV2)</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            '<div class="clean-card" style="text-align:center; padding:1.2rem;">'
            '<span class="badge badge-cyan">Dataset</span>'
            '<h2 style="color:#00BCD4; margin:0.5rem 0 0.2rem 0; font-weight:800;">600</h2>'
            '<p style="color:#64748B; font-size:0.85rem; margin:0;">Citra (Oxford-IIIT Pet)</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

    # Komparasi Ras dalam Gaya Simple
    st.markdown('<div class="section-title">Perbedaan Kunci Ketiga Ras</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Ketiga ras sama-sama memiliki mata biru dan pola colorpoint, tetapi berbeda pada aspek berikut:</div>', unsafe_allow_html=True)

    for breed, info in BREED_INFO.items():
        st.markdown(
            f'<div class="clean-card" style="padding: 1.2rem 1.4rem; margin-bottom: 0.9rem;">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">'
            f'<h4 style="margin:0; color:#0F172A; font-weight:800;">{breed}</h4>'
            f'<span class="badge badge-gray">{info["ciri_khas"].split(",")[0]}</span>'
            f'</div>'
            f'<p style="margin:0; font-size:0.92rem; color:#475569; line-height:1.55;">{info["deskripsi"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

    # Info Teknis Sederhana
    st.markdown(
        '<div class="clean-card" style="border-left: 4px solid #64748B; font-size:0.88rem; color:#64748B;">'
        '<b>Alur Pemrosesan:</b> Citra Input (224×224) ➔ Penskalaan [-1, 1] ➔ Ekstraksi Fitur MobileNetV2 Global Avg Pooling ➔ Standarisasi StandardScaler ➔ Klasifikasi SVM RBF (C=10, γ=scale) ➔ Probabilitas Softmax.'
        '</div>',
        unsafe_allow_html=True,
    )

# ─── Main Program ─────────────────────────────────────────────────────────────
def main():
    inject_css()

    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    navbar()

    page = st.session_state["page"]
    if page == "Home":
        page_home()
    elif page == "Statistik":
        page_statistik()
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
