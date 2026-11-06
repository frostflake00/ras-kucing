# 🐱 Klasifikasi 3 Ras Kucing Serupa (Birman, Ragdoll, Siamese)

> **Proyek Mata Kuliah Pengenalan Pola**  
> Sistem klasifikasi citra cerdas berbasis kombinasi **Deep Learning Feature Extraction (MobileNetV2)** dan **Machine Learning (Support Vector Machine & K-Nearest Neighbors)** untuk membedakan tiga ras kucing dengan kemiripan visual tinggi (*colorpoint pattern*).

---

## 📌 1. Latar Belakang & Masalah
Dalam klasifikasi ras kucing, ras **Birman**, **Ragdoll**, dan **Siamese** merupakan contoh klasik dari permasalahan *fine-grained visual classification*. Ketiga ras ini memiliki kemiripan morfologi yang sangat tinggi:
- Sama-sama memiliki mata berwarna biru cerah.
- Sama-sama memiliki pola warna gelap pada ujung tubuh (*colorpoint* pada wajah, telinga, dan ekor).

Penelitian ini membandingkan kinerja model **K-Nearest Neighbors (KNN)** dan **Support Vector Machine (SVM)** yang dilatih di atas vektor representasi fitur visual tingkat tinggi (*deep features*) berdimensi 1.280 dari arsitektur konvolusi **MobileNetV2** (pre-trained ImageNet).

---

## 🐾 2. Karakteristik Ras Target

| Ras | Ciri Visual Utama | Tipe Tubuh & Bulu |
| :--- | :--- | :--- |
| **Birman** | Memiliki ciri khas sarung tangan putih (*white gloves*) pada keempat telapak kakinya, mata biru safir. | Bulu semi-panjang, tubuh proporsional kokoh. |
| **Ragdoll** | Berbadan besar dan berotot, mata oval biru, tubuh melemas rileks saat digendong (*doll-like*). | Bulu semi-panjang sangat halus seperti kelinci. |
| **Siamese** | Wajah runcing segitiga (*oriental look*), mata berbentuk almond biru tajam, kontras warna point sangat tegas. | Bulu pendek rapat (*short-haired*), tubuh ramping atletis (*slender*). |

---

## 🧠 3. Metodologi & Alur Pemrosesan (Pipeline)

Sistem bekerja melalui arsitektur hibrida (Transfer Learning Feature Extractor + Classical Classifier)

1. **Dataset**: Oxford-IIIT Pet Dataset, difilter secara ketat sebanyak 600 citra seimbang (*balanced dataset*):
   - Birman: 200 citra
   - Ragdoll: 200 citra
   - Siamese: 200 citra
2. **Pembagian Data**: *Stratified Train-Test Split* (80% Train = 480 citra, 20% Test = 120 citra) dengan `random_state=42`.
3. **Ekstraksi Fitur**: MobileNetV2 (*without top*, *global average pooling*) menghasilkan 1.280 fitur per citra.
4. **Penskalaan Fitur**: `StandardScaler` diintegrasikan dalam Scikit-Learn `Pipeline` untuk mencegah *data leakage*.
5. **Hyperparameter Tuning**: `GridSearchCV` dengan 5-Fold Stratified Cross-Validation:
   - **KNN**: Mencari nilai $k \in [3, 5, 7, 9, 11, 15]$, bobot (`uniform`, `distance`), dan metrik jarak (`euclidean`, `manhattan`).
   - **SVM**: Mencari nilai penalti $C \in [0.1, 1, 10, 100]$, kernel (`linear`, `rbf`, `poly`), dan koefisien $\gamma$ (`scale`, `auto`).

---

## 📊 4. Hasil Eksperimen & Evaluasi Model

Evaluasi model pada data uji independen (Test Set: 120 citra, 40 per ras):

| Model | Konfigurasi Hyperparameter Terbaik | Akurasi Uji | Macro F1-Score |
| :--- | :--- | :---: | :---: |
| **Dummy Baseline (Uniform)** | *Random Prior* | 33.3% | 0.333 |
| **K-Nearest Neighbors (KNN)** | $k=7$, `weights='distance'`, `metric='euclidean'` | **80.0%** | **0.805** |
| **Support Vector Machine (SVM)** | `kernel='rbf'`, $C=10$, $\gamma='scale'$ | **87.5%** | **0.876** |

### 🔍 Temuan & Analisis Ilmiah:
1. **Keunggulan SVM atas KNN:**
   Pada ruang berdimensi tinggi ($D = 1280$), SVM terbukti lebih tangguh terhadap fenomena *curse of dimensionality* karena prinsip maksimisasi margin (*maximum margin hyperplane*). Sebaliknya, KNN rentan terhadap saturasi jarak geometris (*distance concentration effect*).
2. **Pola Kesalahan Klasifikasi (*Confusion Pattern*):**
   - **Siamese** paling mudah dikenali dengan akurasi tertinggi karena tekstur bulu pendek dan morfologi ramping yang sangat kontras.
   - **Birman dan Ragdoll** paling sering tertukar (*confused pair*), terutama pada sudut pengambilan foto jarak dekat (*close-up* wajah) di mana sarung tangan putih pada kaki Birman tidak tampak dalam bingkai foto.

## 👥 5. Informasi Pembuat
Proyek ini disusun untuk keperluan akademik mata kuliah **Pengenalan Pola (Pattern Recognition)** Program Studi Informatika / Ilmu Komputer Semester 7.


## 👥 6. akses web
https://ras-kucing-birman-ragdoll-siamese.streamlit.app/