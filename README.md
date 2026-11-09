# Sistem Prediksi IPK Mahasiswa & Explainable AI (SHAP)

Sistem berbasis **Python (`scikit-learn` & `SHAP`)** untuk memprediksi capaian nilai / **IPK akhir Semester 2** dari data awal semester, dilengkapi visualisasi faktor paling berpengaruh menggunakan **SHAP (Bar Plot)** sebagai alat bantu Dosen Pembimbing Akademik (PA).

---

## Spesifikasi & Ruang Lingkup Proyek

| Komponen | Keterangan |
| :--- | :--- |
| **Dataset** | UCI *Students' Dropout and Academic Success* (`4.424` mahasiswa) — publik, bersih, tanpa data sensitif internal. |
| **Target Prediksi** | Nilai rata-rata Semester 2 (`curricular_units_2nd_sem_grade`, skala `0–20` & konversi IPK `0.00–4.00`). |
| **Fitur Input** | **31 fitur awal semester** bebas *target leakage* (data akademik Semester 1, finansial SPP/beasiswa, & pendaftaran). |
| **Model yang Dibandingkan** | **Decision Tree Regressor** vs. **Random Forest Regressor** (5-Fold Cross-Validation Tuning). |
| **Metrik Evaluasi** | `RMSE`, `MAE`, dan `R²`. |
| **Feature Importance** | **SHAP (`TreeExplainer`)** dalam bentuk *Horizontal Bar Plot*. |

---

## Struktur Repositori (Bersih Tanpa Notebook)

```text
├── students_dropout_academic_success.csv   # Dataset mentah UCI (4.424 baris x 37 kolom)
├── preprocess_data.py                      # Pipeline pembersihan data & eliminasi leakage
├── train_and_compare.py                    # Pelatihan & perbandingan Decision Tree vs Random Forest + SHAP
├── predictor.py                            # Modul inferensi 1 mahasiswa: predict_and_explain()
├── generate_proposal_pdf.py                # Skrip pembuat PDF Proposal Bab 1
├── PROPOSAL_BAB_1_KELOMPOK_1.pdf           # Dokumen PDF Proposal Bab 1
├── requirements.txt                        # Daftar pustaka Python
│
├── processed_data/                         # Output dari preprocess_data.py
│   ├── X_train.csv & y_train.csv           # Data latih (3.539 mahasiswa x 31 fitur)
│   ├── X_test.csv  & y_test.csv            # Data uji (885 mahasiswa x 31 fitur)
│   ├── train.csv   & test.csv              # Data gabungan fitur + target
│   └── data_cleaned_unscaled.csv           # Dataset bersih keseluruhan
│
└── model/                                  # Output dari train_and_compare.py
    ├── final_pipeline.joblib               # Pipeline model terbaik (Random Forest)
    ├── model_metadata.json                 # Hasil perbandingan metrik & skema fitur
    ├── model_comparison.csv                # Tabel perbandingan Decision Tree vs Random Forest
    ├── perbandingan_dt_vs_rf.png           # Grafik bar perbandingan RMSE, MAE, & R²
    ├── shap_feature_importance_barplot.png # Grafik SHAP Bar Plot faktor paling berpengaruh
    └── demo_students.json                  # Sampel profil mahasiswa untuk pengujian
```

---

## Cara Menjalankan

### 1. Pembersihan Data (`preprocess_data.py`)
```bash
.venv/bin/python preprocess_data.py
```

### 2. Pelatihan, Perbandingan Model & Grafik SHAP (`train_and_compare.py`)
```bash
.venv/bin/python train_and_compare.py
```

### 3. Uji Prediksi 1 Mahasiswa (`predictor.py`)
```bash
.venv/bin/python predictor.py
```
