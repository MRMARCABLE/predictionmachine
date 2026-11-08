# AGENTS.md — Instruksi Otomatis untuk AI Agent (Proyek P08 Kelompok 1)

Ketika Anda (AI Agent) diminta untuk memindai (*scan*), membaca, atau melanjutkan pekerjaan di repositori ini, **ikuti protokol wajib berikut**:

## 1. Baca `PROJECT_NOTES_P08.md` Terlebih Dahulu
Seluruh perencanaan (*planning*), aturan **Panduan Pengerjaan Proyek P08**, daftar 31 fitur bebas *target leakage*, serta kontrak teknis antar-peran telah didokumentasikan secara lengkap di dalam file [`PROJECT_NOTES_P08.md`](./PROJECT_NOTES_P08.md).

## 2. Protokol Saat Diminta Melakukan *Review / Scan* Folder
Jika pengguna meminta Anda me-*review* atau memindai folder proyek ini:
1. **Jangan mengubah kode apa pun** sebelum pengguna memilih tugas yang ingin dikerjakan.
2. **Langsung berikan jawaban terstruktur sesuai *planning* di `PROJECT_NOTES_P08.md`**, meliputi:
   - **Status Peran 1 — Data Engineer (`preprocess_data.py` & `processed_data/`)**:
     Jelaskan apa yang sudah ada dan **5 poin perbaikan wajib** sesuai Bab 3 Panduan P08 (yaitu: sinkronisasi target ke Regresi `curricular_units_2nd_sem_grade`, penghapusan fitur Semester 2 penyebab *leakage*, penghapusan `StandardScaler` & `OneHotEncoder` dari tahap Data Engineer karena sudah ditangani di Pipeline ML Engineer, pembuatan Notebook EDA `eda_data_engineer.ipynb`, serta pembuatan `DATA_DICTIONARY.md`).
   - **Status Peran 2 — ML Engineer (`ml_engineer_regression.ipynb` & `predictor.py`)**:
     Jelaskan bahwa pipeline Regresi (`DummyRegressor`, `DecisionTreeRegressor`, `RandomForestRegressor` dengan 5-Fold CV tuning, *Learning Curve*, Plot Prediksi vs Aktual, Plot Residual, SHAP `TreeExplainer` dengan agregasi *one-hot*, dan fungsi `predict_and_explain()`) **sudah siap 100%** dan hanya perlu dieksekusi (*Run All*) dengan kernel `.venv` agar menghasilkan folder `model/` (`final_pipeline.joblib`, `model_metadata.json`, `demo_students.json`).
   - **Status Peran 3 — App Developer (`app.py`)**:
     Jelaskan bahwa aplikasi Streamlit (`app.py`) belum dibuat dan nantinya wajib memanggil `predict_and_explain(data_satu_mahasiswa)` dari `predictor.py` serta memuat sampel tombol *"Isi Contoh"* dari `model/demo_students.json`.
3. **Tawarkan Langkah Eksekusi Berikutnya (Prioritas 1, 2, atau 3)**:
   Tanyakan langsung kepada pengguna bagian mana dari **Bagian 5 (`Daftar Tugas Selanjutnya`) di `PROJECT_NOTES_P08.md`** yang ingin dikerjakan sekarang.

## 3. Aturan Teknis yang Tidak Boleh Dilanggar
- **Konsistensi**: Gunakan selalu `RANDOM_STATE = 42` dan versi paket yang terkunci di `requirements.txt`.
- **Pencegahan *Data/Target Leakage***: Jangan pernah memasukkan `target`, `curricular_units_2nd_sem_credited`, `curricular_units_2nd_sem_enrolled`, `curricular_units_2nd_sem_evaluations`, `curricular_units_2nd_sem_approved`, atau `curricular_units_2nd_sem_without_evaluations` ke dalam fitur input (`X`) saat memprediksi `curricular_units_2nd_sem_grade`.
- **Tanpa `StandardScaler` pada Model Pohon**: Decision Tree dan Random Forest tidak membutuhkan normalisasi; transformasi `OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)` wajib berada di dalam `ColumnTransformer` milik `Pipeline` model.

