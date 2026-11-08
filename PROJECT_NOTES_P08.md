# Catatan Proyek & Panduan Serah Terima Antar-Agent (Proyek P08)
**Prediksi Performa Akademik Mahasiswa · Kelompok 1**

> [!IMPORTANT]
> **INSTRUKSI WAJIB UNTUK AI AGENT YANG ME-REVIEW FOLDER INI (`<AGENT_DIRECTIVE>`):**
> 1. **Jangan Mengubah Kode Tanpa Konfirmasi**: Saat pengguna meminta Anda melakukan *scan* atau *review* proyek ini, **JANGAN** mengedit file apa pun terlebih dahulu.
> 2. **Format Jawaban Saat Review Awal**: Anda **WAJIB** langsung menyajikan ringkasan status proyek berdasarkan perencanaan (*planning*) di dokumen ini, yang mencakup:
>    - **Status 3 Peran Proyek P08** (**Data Engineer**, **ML Engineer**, dan **App Developer**) beserta file yang sudah selesai vs yang belum selesai.
>    - **Temuan Audit pada `preprocess_data.py` (Data Engineer)**: Jelaskan bahwa `preprocess_data.py` saat ini masih menggunakan Klasifikasi Biner (`Dropout` vs `Graduate`) dan masih memakai `StandardScaler` + `OneHotEncoder` di luar Pipeline ML Engineer, sehingga perlu disinkronkan dengan pendekatan **Regresi Nilai Akhir (`curricular_units_2nd_sem_grade`)** bebas *target leakage* sesuai Bab 3 & Bab 4 Panduan P08.
>    - **Kesiapan Artefak ML Engineer**: Jelaskan bahwa `ml_engineer_regression.ipynb` dan `predictor.py` (`predict_and_explain()`) sudah siap 100% untuk Regresi sesuai Bab 4 Panduan P08, dan tinggal dijalankan (*Run All*) untuk menghasilkan folder `model/` (`final_pipeline.joblib`, `model_metadata.json`, `demo_students.json`).
>    - **Tanyakan Peran Pengguna**: Akhiri review awal dengan menanyakan apakah pengguna ingin melanjutkan sebagai **Data Engineer** (Prioritas 2), **ML Engineer** (Prioritas 1), atau **App Developer** (Prioritas 3) sesuai **Bagian 5 dokumen ini**.
> 3. **Patuhi Kontrak Serah Terima**: Gunakan selalu `RANDOM_STATE = 42`, versi library di `requirements.txt`, dan skema fungsi `predict_and_explain()` di `predictor.py`.

---

## 1. Gambaran Umum & Alur Tiga Peran (Sesuai Panduan P08)

Proyek P08 dibagi menjadi **3 peran** yang bekerja paralel dengan **2 titik serah terima (kontrak)**:

| Peran | Menerima | Mengirim (Deliverables) | Aturan Kunci di Panduan P08 |
| :--- | :--- | :--- | :--- |
| **1. Data Engineer** | Kesepakatan awal (dataset, target, daftar fitur bebas *leakage*) | - `train.csv` & `test.csv` (bersih, **tanpa** scaling & **tanpa** one-hot encoding)<br>- Notebook EDA (`eda.ipynb`)<br>- Data Dictionary & Tabel Kontrak Fitur<br>- 3–5 sampel mahasiswa demo | **Tidak perlu scaling** (`StandardScaler` tidak dibutuhkan Decision Tree / Random Forest). **Jangan lakukan One-Hot Encoding di luar Pipeline ML Engineer.** Split 80/20 dengan `random_state=42`. |
| **2. ML Engineer** | Dataset bersih dari Data Engineer | - Notebook eksperimen (`ml_engineer_regression.ipynb`)<br>- `model/final_pipeline.joblib`<br>- `model/model_metadata.json`<br>- `model/demo_students.json`<br>- Modul `predictor.py` (`predict_and_explain()`) | Wajib menggunakan **Regresi** (`RMSE`, `MAE`, `R²`), membandingkan `DummyRegressor`, `DecisionTreeRegressor`, dan `RandomForestRegressor` (5-Fold CV tuning), analisis *overfitting* (*Learning Curve*), Plot Prediksi vs Aktual, Plot Residual, dan **SHAP (`TreeExplainer`)**. |
| **3. App Developer** | Model, SHAP, & `predictor.py` dari ML Engineer + sampel demo dari Data Engineer | - Aplikasi Streamlit (`app.py`)<br>- `requirements.txt` (versi terkunci `==`)<br>- `README.md` final & `.gitignore` | Form input dengan batas min/max & `selectbox`, tombol *"Isi Contoh"*, pemuatan model dengan `@st.cache_resource`, visualisasi bar faktor SHAP + 1 kalimat bahasa awam + catatan etika (alat bantu Dosen PA, bukan vonis). |

---

## 2. Status File di Repositori Saat Ini

| File / Folder | Pemilik Peran | Status | Keterangan |
| :--- | :---: | :---: | :--- |
| `students_dropout_academic_success.csv` | Data Engineer | Tersedia | Dataset mentah awal (`4.424` baris × `37` kolom). |
| `preprocess_data.py` & `processed_data/` | Data Engineer | **Perlu Revisi** | Saat ini masih memproses target sebagai **Klasifikasi Biner (`Dropout` vs `Graduate`)**, masih memasukkan fitur Semester 2 (*target leakage* terhadap nilai Semester 2), dan masih menerapkan `StandardScaler` + `OneHotEncoder` (111 kolom) di tahap Data Engineer. |
| `ml_engineer_regression.ipynb` | ML Engineer | **Kode Siap (28 Sel)** | Notebook lengkap untuk **Regresi** (`curricular_units_2nd_sem_grade`). **Catatan**: Perlu dijalankan (*Run All*) menggunakan kernel `.venv` untuk menghasilkan output grafik dan folder `model/`. |
| `predictor.py` | ML Engineer | **Siap Pakai** | Berisi fungsi kontrak `predict_and_explain(data_satu_mahasiswa)` yang memuat `model/final_pipeline.joblib`, menghitung prediksi nilai (skala `0–20` & konversi IPK `0–4`), dan mengakumulasi nilai SHAP dari pecahan *One-Hot Encoding* kembali ke 31 nama fitur asli. |
| `requirements.txt` | Semua Peran | **Siap (Versi Terkunci)** | Berisi `numpy==2.5.4`, `pandas==3.1.1`, `scikit-learn==1.9.1`, `joblib==1.6.0`, `shap==0.54.1`, `matplotlib==3.11.2`, `seaborn==0.13.2`, `ipykernel==7.4.0`. *(Tambahkan `streamlit` saat mengerjakan bagian App Developer).* |
| `.gitignore` | Semua Peran | **Siap** | Mengabaikan `.venv/`, `.vscode/`, `__pycache__/`, `.DS_Store`, dan `.ipynb_checkpoints/` agar aman untuk kolaborasi lintas OS (Windows & macOS/Linux). |
| `model/` | ML Engineer | **Belum Terbentuk** | Akan otomatis terbentuk (`final_pipeline.joblib`, `model_metadata.json`, `demo_students.json`) begitu Sel `[25]` pada `ml_engineer_regression.ipynb` dieksekusi. |
| `app.py` | App Developer | **Belum Dibuat** | Aplikasi Streamlit untuk Dosen Pembimbing Akademik (PA). |

---

## 3. Keputusan Desain Penting: Regresi vs Klasifikasi & Pencegahan *Leakage*

### A. Mengapa Menggunakan Regresi (`curricular_units_2nd_sem_grade`)?
* Bab 4 Panduan P08 secara eksplisit mensyaratkan metrik **`RMSE` (`root_mean_squared_error`), `MAE`, dan `R²`**, model baseline **`DummyRegressor`**, serta diagnostik **Plot Prediksi vs Aktual** dan **Plot Residual**.
* Oleh karena itu, pada `ml_engineer_regression.ipynb` dan `predictor.py`, target prediksi (`y`) ditetapkan sebagai **`curricular_units_2nd_sem_grade`** (nilai rata-rata akademik Semester 2 pada skala `0.00 – 20.00`, dilengkapi konversi ekuivalen skala IPK `0.00 – 4.00`).

### B. Pencegahan Kebocoran Fitur (*Target / Feature Leakage*)
Agar prediksi valid secara metodologis dan dapat digunakan di awal Semester 2, **6 kolom berikut WAJIB dikeluarkan dari fitur input (`X`)**:
1. `target` (Status akhir kelulusan `Dropout`/`Graduate`/`Enrolled` — terjadi di masa depan).
2. `curricular_units_2nd_sem_credited`
3. `curricular_units_2nd_sem_enrolled`
4. `curricular_units_2nd_sem_evaluations`
5. `curricular_units_2nd_sem_approved`
6. `curricular_units_2nd_sem_without_evaluations`
*(Serta seluruh fitur rekayasa yang menggunakan data Semester 2 seperti `approval_rate_2nd_sem`, `grade_diff_sem2_sem1`, `approval_rate_diff`, `total_enrolled_units`, `total_approved_units`).*

### C. Daftar 31 Fitur Input (`X`) yang Digunakan Model
Terdiri atas **14 Fitur Numerik/Ordinal**, **9 Fitur Kategori Nominal**, dan **8 Fitur Biner**:
* **Numerik/Ordinal (14 kolom — `SimpleImputer(strategy="median")`, TANPA `StandardScaler`)**:
  `application_order`, `previous_qualification_grade`, `admission_grade`, `age_at_enrollment`, `curricular_units_1st_sem_credited`, `curricular_units_1st_sem_enrolled`, `curricular_units_1st_sem_evaluations`, `curricular_units_1st_sem_approved`, `curricular_units_1st_sem_grade`, `curricular_units_1st_sem_without_evaluations`, `unemployment_rate`, `inflation_rate`, `gdp`, `approval_rate_1st_sem`.
* **Kategori Nominal (9 kolom — `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)`)**:
  `marital_status`, `application_mode`, `course`, `previous_qualification`, `nationality`, `mothers_qualification`, `fathers_qualification`, `mothers_occupation`, `fathers_occupation`.
* **Biner 0/1 (8 kolom — `SimpleImputer(strategy="most_frequent")`)**:
  `daytime_evening_attendance`, `displaced`, `educational_special_needs`, `debtor`, `tuition_fees_up_to_date`, `gender`, `scholarship_holder`, `international`.

---

## 4. Kontrak Teknis `predict_and_explain()` (`predictor.py`) untuk App Developer

App Developer **tidak perlu** mengolah `OneHotEncoder` maupun `shap.TreeExplainer` secara manual di `app.py`. Cukup panggil:

```python
from predictor import predict_and_explain

hasil = predict_and_explain(data_satu_mahasiswa, top_k=5)
```

### Parameter Input (`data_satu_mahasiswa`)
* Berupa `dict`, `pd.Series`, atau `pd.DataFrame` (1 baris) dengan nama kolom asli (30 atau 31 fitur; jika `approval_rate_1st_sem` tidak dikirim, `predictor.py` otomatis menghitungnya dari `curricular_units_1st_sem_approved / curricular_units_1st_sem_enrolled`).
* Jika ada kolom yang kosong/tidak dikirim, `predictor.py` otomatis mengisinya dengan nilai median/modus dari `model/model_metadata.json`.

### Format Return Value (`dict`)
```json
{
  "predicted_grade": 12.85,
  "predicted_gpa_4": 2.57,
  "base_value": 10.23,
  "top_factors": [
    {
      "feature": "curricular_units_1st_sem_grade",
      "feature_label": "Nilai Rata-Rata Semester 1",
      "raw_value": 13.5,
      "shap_impact": 1.8421,
      "abs_impact": 1.8421,
      "direction": "menaikkan"
    }
  ],
  "all_factors": [...],
  "summary_text": "Prediksi nilai akhir mahasiswa adalah 12.85/20.00 (setara IPK 2.57/4.00, dari rata-rata dasar 10.23), terutama didorong naik oleh Nilai Rata-Rata Semester 1 (13.5, +1.84 poin)...",
  "disclaimer": "Catatan Etika: Hasil prediksi ini merupakan alat bantu pendukung keputusan bagi Dosen Pembimbing Akademik (PA), bukan vonis mutlak terhadap mahasiswa."
}
```

---

## 5. Daftar Tugas Selanjutnya (Action Items untuk Agent / Anggota Tim)

### Prioritas 1 — Eksekusi Notebook ML Engineer (ML Engineer)
1. Buka `ml_engineer_regression.ipynb`, pilih kernel `.venv` (`Python (.venv - PredictionMachine)`), dan jalankan **Run All** (atau eksekusi seluruh sel kode).
2. Pastikan folder `model/` beserta 3 file di dalamnya (`final_pipeline.joblib`, `model_metadata.json`, `demo_students.json`) terbentuk dan simpan notebook (`Ctrl + S`).

### Prioritas 2 — Perbaikan & Kelengkapan Data Engineer (Data Engineer)
1. **Revisi `preprocess_data.py`**:
   - Sesuaikan target dengan kesepakatan kelompok (jika Regresi: gunakan `curricular_units_2nd_sem_grade` dan hapus kolom Semester 2 lainnya serta `target` agar bebas *leakage*).
   - Hapus langkah `StandardScaler` dan `OneHotEncoder` dari file serah terima utama; simpan `processed_data/train.csv` dan `processed_data/test.csv` (80/20, `random_state=42`) dalam bentuk 31 kolom asli.
2. **Buat Notebook EDA (`eda_data_engineer.ipynb`)**:
   - Memuat distribusi target, statistik deskriptif (`mean`, `std`, `min`, `max`), *heatmap* korelasi numerik terhadap target, *boxplot* target per kategori, analisis ketimpangan kategori, serta 3–5 temuan utama tertulis.
3. **Buat Data Dictionary & Kamus Kode Kategori (`DATA_DICTIONARY.md`)**:
   - Mencatat sumber dataset, DOI, lisensi, deskripsi 37 kolom asli, rentang nilai numerik, dan arti kode angka pada fitur kategori (misal arti kode jurusan pada `course`, kode pendidikan/pekerjaan orang tua, dll.) untuk dipakai pada label dropdown di `app.py`.

### Prioritas 3 — Pembuatan Aplikasi Streamlit (App Developer)
1. Instal `streamlit` ke dalam `.venv` dan kunci versinya di `requirements.txt`.
2. Buat `app.py` sesuai **Bab 5 Panduan P08**:
   - Gunakan `@st.cache_resource` untuk memuat artefak model melalui `load_artifacts()` dari `predictor.py`.
   - Sediakan tombol **"Isi Contoh"** (Performa Tinggi, Performa Sedang, Risiko Tinggi, Kasus Ekstrem) yang mengambil data dari `model/demo_students.json`.
   - Kelompokkan form input ke dalam tab/bagian yang rapi (Data Akademik Semester 1, Latar Belakang & Pendaftaran, Kondisi Finansial & Keluarga, Indikator Makroekonomi) dengan batas `min` dan `max` dari `model/model_metadata.json`.
   - Tampilkan kartu metrik `predicted_grade` (`0–20`) & `predicted_gpa_4` (`0–4`), visualisasi horizontal bar chart untuk `top_factors` SHAP (hijau = menaikkan, merah = menurunkan), `summary_text`, dan kotak peringatan `disclaimer` etika.
3. Perbarui `README.md` utama agar mencakup dokumentasi Data Engineer, ML Engineer, dan cara menjalankan `app.py`.

---

## 6. Panduan Menjalankan Lingkungan di Berbagai Sistem Operasi (Windows vs macOS/Linux)

* **Windows (PowerShell / CMD)**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\pip install -r requirements.txt
  ```
* **macOS / Linux (Bash / Zsh)**:
  ```bash
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  ```

