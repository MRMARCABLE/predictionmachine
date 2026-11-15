# Sistem Preskriptif Deteksi Dini Risiko Drop-Out Mahasiswa Menggunakan Random Forest dan SHAP Explainability

Dokumen ini berfungsi sebagai **Catatan Progres & Panduan Handover Tim** setelah dilakukan perombakan ulang (*reset* pola pikir dan alur kerja) agar proyek bebas dari *data leakage* dan mengikuti arsitektur *machine learning* yang benar.

---

## Ringkasan Perubahan Pola Pikir (Kenapa Diulang dari Awal?)

1. **Perubahan Target & Jenis Masalah:**
   * **Sebelumnya (Keliru):** Regresi memprediksi nilai Semester 2 dan mempertahankan baris mahasiswa berstatus `Enrolled`.
   * **Sekarang (Benar):** **Klasifikasi Biner Tegas (`Dropout = 1` vs `Graduate = 0`)**. Baris dengan status `Enrolled` (`794` baris) di-*drop* karena mahasiswa tersebut masih aktif kuliah dan belum memiliki hasil akhir.
2. **Pencegahan *Data Leakage* pada Preprocessing & SMOTE:**
   * **Urutan Wajib:** Pemisahan `train_test_split` dilakukan **TERLEBIH DAHULU** sebelum *One-Hot Encoding*, *StandardScaler*, dan **SMOTE**.
   * `OneHotEncoder` dan `StandardScaler` hanya di-*fit* pada `X_train`.
   * Algoritma **SMOTE** hanya diterapkan pada **data training (`X_train_encoded`, `y_train`)**. Data testing (`X_test_encoded`, `y_test`) dibiarkan **100% murni/asli** agar evaluasi model tidak bias atau *overfitting*.

---

## BAGIAN 1: APA YANG SUDAH DILAKUKAN (SELESAI ✅)

### ✅ Tahap 1: Eksplorasi & Pemisahan Domain Fitur (EDA)
* **File Eksekusi:** [`preprocess_data.py`](preprocess_data.py) (fungsi `load_and_prepare_eda()`)
* **Dataset:** [`students_dropout_academic_success.csv`](students_dropout_academic_success.csv) (`4.424` baris awal $\rightarrow$ setelah *drop* `Enrolled` menjadi **`3.630` baris × `37` kolom**, `0` *missing values*).
* **Distribusi Target (`Target_Binary`):**
  * `0` (`Graduate`) : **2.209 mahasiswa (60,85%)**
  * `1` (`Dropout`)  : **1.421 mahasiswa (39,15%)**
* **Pemisahan 37 Kolom (`36` Fitur Independen $X$ + `1` Kolom Target $y$):**
  1. **Domain Demografi (`6` fitur):** `Marital status`, `Nacionality`, `Displaced`, `Gender`, `Age at enrollment`, `International`
  2. **Domain Sosial-Ekonomi & Finansial (`11` fitur):** `Mother's qualification`, `Father's qualification`, `Mother's occupation`, `Father's occupation`, `Educational special needs`, `Debtor`, `Tuition fees up to date`, `Scholarship holder`, `Unemployment rate`, `Inflation rate`, `GDP`
  3. **Domain Akademik Awal / Admisi (`7` fitur):** `Application mode`, `Application order`, `Course`, `Daytime/evening attendance`, `Previous qualification`, `Previous qualification (grade)`, `Admission grade`
  4. **Domain Performa Akademik Semester 1 (`6` fitur):** `Curricular units 1st sem (credited, enrolled, evaluations, approved, grade, without evaluations)`
  5. **Domain Performa Akademik Semester 2 (`6` fitur):** `Curricular units 2nd sem (credited, enrolled, evaluations, approved, grade, without evaluations)`
  6. **Target Output (`1` kolom):** `Target_Binary` (`1 = Dropout`, `0 = Graduate`)

---

### ✅ Tahap 2: Preprocessing & Pemisahan Data (Anti-Data Leakage)
* **File Eksekusi:** [`preprocess_data.py`](preprocess_data.py) (fungsi `preprocess_and_split_no_leakage()`)
* **Detail Perlakuan 36 Fitur Input ($X$):**
  * **9 Fitur Kategorikal Nominal** (`Marital status`, `Application mode`, `Course`, `Previous qualification`, `Nacionality`, `Mother's/Father's qualification & occupation`) $\rightarrow$ `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`
  * **8 Fitur Biner (0/1)** (`Daytime/evening attendance`, `Displaced`, `Educational special needs`, `Debtor`, `Tuition fees up to date`, `Gender`, `Scholarship holder`, `International`) $\rightarrow$ `passthrough`
  * **19 Fitur Numerik** $\rightarrow$ `StandardScaler()`
* **Hasil Transformasi & Balancing SMOTE:**
  * **Train Set Sebelum SMOTE (`X_train_encoded`):** `2.904 baris × 227 kolom` (`Graduate = 1.767`, `Dropout = 1.137`)
  * **Train Set Sesudah SMOTE (`X_train_smote`):** `3.534 baris × 227 kolom` (**Seimbang 50:50** $\rightarrow$ `Graduate = 1.767`, `Dropout = 1.767`)
  * **Test Set Murni (`X_test_encoded`):** `726 baris × 227 kolom` (**Tidak di-SMOTE** $\rightarrow$ `Graduate = 442`, `Dropout = 284`)
* **Output Tersimpan di [`processed_data/`](processed_data/):**
  * `X_train_smote.csv` & `y_train_smote.csv` (Dipakai untuk *training* model)
  * `X_test_encoded.csv` & `y_test.csv` (Dipakai untuk evaluasi *test set* murni)
  * `X_train_encoded.csv` & `y_train.csv` (Data latih terenkode sebelum SMOTE)
  * `X_train_raw.csv` & `X_test_raw.csv` (Data asli 36 fitur sebelum *One-Hot Encoding*)
  * `preprocessor.joblib` (Pipeline `ColumnTransformer` yang sudah di-*fit* pada `X_train`)

---

### ✅ Tahap 3: Training Model, Evaluasi & Serialisasi (`Scikit-Learn` & `Joblib`)
* **File Eksekusi:** [`train_and_compare.py`](train_and_compare.py)
* **Proses yang Dilakukan:**
  1. Melatih **Decision Tree Classifier** sebagai *baseline* pada data `X_train_smote`.
  2. Melatih & mengoptimasi **Random Forest Classifier** menggunakan `GridSearchCV` (5-Fold Stratified Cross-Validation, optimasi skor `f1`).
  3. **Parameter Terbaik Random Forest:** `{'n_estimators': 300, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}` (Skor F1 CV: `0.9133`).
  4. Mengevaluasi kedua model pada **Test Set Murni (`726` baris, tidak di-SMOTE)**:

| Metrik Evaluasi (Test Set Murni) | Decision Tree (Baseline) | Random Forest (Optimal) |
| :--- | :---: | :---: |
| **Precision (`Dropout = 1`)** | 79,05% (`0.7905`) | **91,30% (`0.9130`)** |
| **Recall (`Dropout = 1`)** | 87,68% (`0.8768`) | **88,73% (`0.8873`)** |
| **F1-Score (`Dropout = 1`)** | 83,14% (`0.8314`) | **90,00% (`0.9000`)** |
| **Accuracy** | 86,09% (`0.8609`) | **92,29% (`0.9229`)** |
| **ROC-AUC** | `0.8637` | **`0.9724`** |

* **Output Tersimpan di [`model/`](model/):**
  * `model/random_forest_dropout.joblib` — Model Random Forest terbaik.
  * `model/decision_tree_baseline.joblib` — Model Decision Tree *baseline*.
  * `model/dropout_classifier_bundle.joblib` — Bundel lengkap (`model`, `baseline_model`, `preprocessor`, `feature_names`, `best_params`, `metrics`).
  * `model/evaluation_metrics.json` — Ringkasan metrik evaluasi dalam format JSON.
  * `model/model_comparison.png` — Visualisasi perbandingan metrik & *Confusion Matrix*.

---

## BAGIAN 2: APA YANG AKAN DILAKUKAN SELANJUTNYA (TUGAS LANJUTAN TIM 📋)

Berikut adalah **4 tahap selanjutnya** yang harus dikerjakan secara berurutan oleh rekan tim:

### 📌 Tahap 4: Integrasi Mesin Penjelas (SHAP)
* **Target File:** Buat/perbarui [`predictor.py`](predictor.py) *(Catatan: file `predictor.py` yang ada saat ini masih versi regresi lama dan perlu ditulis ulang)*.
* **Spesifikasi Teknis:**
  1. Load `model/random_forest_dropout.joblib` dan `processed_data/preprocessor.joblib` (atau langsung load `model/dropout_classifier_bundle.joblib`).
  2. Inisialisasi `explainer = shap.TreeExplainer(rf_model)` khusus untuk arsitektur berbasis pohon (*tree-based*).
  3. Buat fungsi inferensi (misal `predict_single_student(student_dict: dict)`) yang menerima **1 baris data mahasiswa** (36 fitur mentah):
     * Transformasikan input 1 baris tersebut menggunakan `preprocessor.transform(df_input)` menjadi 227 kolom terenkode.
     * Hitung probabilitas prediksi *Dropout* (`rf_model.predict_proba(X_encoded)[0, 1]`).
     * Hitung nilai SHAP (`shap_values = explainer.shap_values(X_encoded)` untuk kelas positif `Dropout = 1`).
     * **Penting (Agregasi Fitur OHE):** Karena fitur kategorikal seperti `Course` dipecah menjadi beberapa kolom One-Hot (misal `Course_171`, `Course_9254`), jumlahkan nilai SHAP dari sub-kolom One-Hot yang berasal dari fitur induk yang sama agar kembali ke **36 nama fitur asli**.
     * Urutkan berdasarkan nilai absolut SHAP tertinggi (`|SHAP|`) dan keluarkan **daftar 3 fitur teratas (Top 3 SHAP Features)** beserta nilai asli fitur dan nilai kontribusi SHAP-nya.

---

### 📌 Tahap 5: Pembangunan Rule-Engine Rekomendasi (Logika Bisnis)
* **Target File:** Di dalam [`predictor.py`](predictor.py) atau modul khusus `recommendation_engine.py`.
* **Spesifikasi Teknis:**
  1. Buat fungsi *mapping* sederhana (menggunakan blok `if-else` atau `dictionary`) yang menerima:
     * Probabilitas prediksi *Dropout* (`prob_dropout`)
     * Daftar **3 fitur teratas dari SHAP** beserta nilai aktual mahasiswa pada fitur tersebut.
  2. Terjemahkan kondisi fitur + tingkat risiko menjadi **kalimat aksi intervensi konkret**:
     * **Contoh Aturan Finansial:**
       * Jika fitur penyumbang masuk Top 3 adalah `Debtor == 1` dan model memprediksi probabilitas DO `> 70%` (`0.70`), kembalikan string:
         > `"Rekomendasi: Jadwalkan pertemuan dengan bagian keuangan untuk restrukturisasi pembayaran."`
       * Jika `Tuition fees up to date == 0`, kembalikan rekomendasi pengingat tunggakan UKT / bantuan finansial darurat.
       * Jika `Scholarship holder == 0` dengan risiko ekonomi tinggi, rekomendasikan pengajuan beasiswa internal.
     * **Contoh Aturan Akademik (`Curricular units 1st/2nd sem approved / grade` rendah):**
       * Rekomendasikan *academic mentoring*, konseling Dosen Pembimbing Akademik (PA), atau evaluasi beban SKS.
     * **Contoh Aturan Demografi/Admisi (`Age at enrollment`, `Displaced`, `Admission grade`):**
       * Rekomendasikan program adaptasi mahasiswa baru / kelas matrikulasi.

---

### 📌 Tahap 6: Pembuatan API Backend (Decoupled Architecture)
* **Target File:** Buat `api.py` (menggunakan **FastAPI** atau **Flask** di Python, atau `net/http` di Go).
* **Prinsip Arsitektur:** **Jangan gabungkan logika model dengan kode antarmuka (UI).** Frontend hanya boleh berkomunikasi dengan model melalui HTTP API.
* **Spesifikasi Endpoint:**
  * **Endpoint:** `POST /predict`
  * **Request Body (JSON):** Berisi 36 parameter fitur dari 1 mahasiswa.
  * **Proses Internal:** Memvalidasi payload $\rightarrow$ memanggil fungsi prediksi + `shap.TreeExplainer` (Tahap 4) $\rightarrow$ memanggil *Rule-Engine* Rekomendasi (Tahap 5).
  * **Response Body (JSON):**
    ```json
    {
      "status": "success",
      "prediction": {
        "label": "Dropout",
        "label_code": 1,
        "probability_dropout": 0.842,
        "probability_graduate": 0.158,
        "risk_level": "HIGH"
      },
      "top_3_shap_features": [
        {
          "feature_name": "Tuition fees up to date",
          "raw_value": 0,
          "shap_value": 0.215,
          "abs_shap_value": 0.215
        },
        {
          "feature_name": "Debtor",
          "raw_value": 1,
          "shap_value": 0.164,
          "abs_shap_value": 0.164
        },
        {
          "feature_name": "Curricular units 2nd sem (approved)",
          "raw_value": 1,
          "shap_value": 0.142,
          "abs_shap_value": 0.142
        }
      ],
      "recommendations": [
        "Rekomendasi: Jadwalkan pertemuan dengan bagian keuangan untuk restrukturisasi pembayaran."
      ]
    }
    ```

---

### 📌 Tahap 7: Pembangunan Frontend (Streamlit)
* **Target File:** Buat `app.py` (Dashboard Streamlit).
* **Spesifikasi Antarmuka & Alur Kerja:**
  1. Desain *form input* (`st.form`) yang bersih dan terkelompok berdasarkan domain fitur (Demografi, Sosial-Ekonomi & Finansial, Admisi, dan Performa Akademik) untuk memasukkan parameter mahasiswa.
  2. Saat tombol **Submit / Prediksi** ditekan, Streamlit melakukan **HTTP Request (`requests.post`)** ke endpoint `POST /predict` pada API Backend (Tahap 6).
  3. Tangkap JSON balasan dari API, lalu tampilkan 3 komponen visualisasi utama:
     * **Gauge Chart:** Menampilkan persentase probabilitas *Dropout* (`0% – 100%`) dengan indikator warna risiko (Hijau / Kuning / Merah).
     * **Horizontal Bar Plot SHAP:** Menampilkan 3 fitur paling berpengaruh dari SHAP beserta arah kontribusinya.
     * **Teks Rekomendasi Intervensi:** Mencetak kalimat rekomendasi aksi dari *Rule-Engine* dengan **format cetak tebal (`**Rekomendasi: ...**`)**.

---

## Cara Menjalankan Pipeline yang Sudah Ada Saat Ini

```bash
# 1. Jalankan EDA, Pemisahan Fitur, One-Hot Encoding, Train-Test Split & SMOTE
.venv/bin/python preprocess_data.py

# 2. Jalankan Training Decision Tree vs Random Forest, Evaluasi Test Murni & Serialisasi Model
.venv/bin/python train_and_compare.py
```
