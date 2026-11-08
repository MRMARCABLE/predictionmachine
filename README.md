# Data Preprocessing & Cleaning — Predict Students' Dropout and Academic Success

Repositori ini berisi pipeline **Data Engineering / Data Preprocessing** untuk membersihkan dan menyiapkan dataset [Predict Students' Dropout and Academic Success](https://www.kaggle.com/datasets/adilshamim8/predict-students-dropout-and-academic-success) agar siap digunakan oleh model Machine Learning (**Binary Classification: Dropout vs Graduate**).

Seluruh proses terbagi menjadi **6 bagian fungsi di dalam `preprocess_data.py`** lengkap dengan catatan (*notes*) di setiap bagiannya agar mudah di-*tracking* dan dipelajari.

---

## Struktur Bagian di `preprocess_data.py`

| Bagian | Fungsi di `preprocess_data.py` | Penjelasan Singkat |
| :---: | :--- | :--- |
| **Bagian A** | `load_and_inspect()` | Membaca dataset mentah (`4.424 × 37`), memeriksa *missing values* (`0`), duplikat (`0`), dan distribusi kelas awal. |
| **Bagian B** | `clean_and_filter_data()` | Standarisasi nama kolom ke `snake_case`, memperbaiki typo `Nacionality`, membuang status `Enrolled` (fokus ke `Graduate = 0` vs `Dropout = 1`, total `3.630` baris), dan mengoreksi anomali `application_order == 0` menjadi `1`. |
| **Bagian C** | `engineer_features()` | Membuat 6 fitur baru (`approval_rate_1st_sem`, `approval_rate_2nd_sem`, `grade_diff_sem2_sem1`, `approval_rate_diff`, `total_enrolled_units`, `total_approved_units`) dan menyimpan `data_cleaned_unscaled.csv`. |
| **Bagian D** | `split_train_test()` | Membagi data menjadi **80% Train (`2.904` baris)** dan **20% Test (`726` baris)** secara *stratified* sebelum menghitung statistik apa pun guna mencegah **Data Leakage**. |
| **Bagian E** | `cap_outliers()` | Melakukan *Winsorizing / Capping* pada nilai ekstrem (`age_at_enrollment`, `admission_grade`, `previous_qualification_grade`) menggunakan batas persentil ke-1 dan ke-99 dari `X_train`. |
| **Bagian F** | `encode_scale_and_save()` | Memisahkan perlakuan untuk **25 fitur Numerik** (`StandardScaler`), **9 fitur Kategori Nominal** (`OneHotEncoder(min_frequency=0.01)`), dan **8 fitur Biner** (`0`/`1`), lalu menyimpan hasil akhir ke `processed_data/`. |

---

## Struktur Output (`processed_data/`)

| File | Deskripsi | Ukuran |
| :--- | :--- | :--- |
| `data_cleaned_unscaled.csv` | Dataset bersih + 6 fitur baru sebelum *One-Hot Encoding* & *StandardScaler* (cocok untuk EDA/visualisasi). | `3.630 × 43` |
| `X_train.csv` | Fitur data latih yang sudah di-*encode* dan di-*scale*. | `2.904 × 111` |
| `y_train.csv` | Target data latih (`0` = Graduate, `1` = Dropout). | `2.904 × 1` |
| `X_test.csv` | Fitur data uji yang sudah di-*encode* dan di-*scale*. | `726 × 111` |
| `y_test.csv` | Target data uji (`0` = Graduate, `1` = Dropout). | `726 × 1` |
| `preprocessor.joblib` | Pipeline `ColumnTransformer` (`scikit-learn`) yang sudah di-*fit* pada data latih. | Binary Object |

---

## Cara Menjalankan

```bash
.venv/bin/python3 preprocess_data.py
```
