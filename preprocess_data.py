"""
================================================================================
PIPELINE DATA PREPROCESSING — PREDICT STUDENTS' DROPOUT VS GRADUATE
================================================================================
File ini dibagi menjadi beberapa bagian fungsi berdasarkan tugasnya agar mudah
di-tracking, dibaca, dan dipelajari:

  1. load_and_inspect()      -> Membaca CSV mentah & inspeksi awal kondisi data
  2. clean_and_filter_data() -> Standarisasi kolom, filter Dropout vs Graduate, & koreksi anomali
  3. engineer_features()     -> Membuat 6 fitur baru (rasio kelulusan & selisih nilai)
  4. split_train_test()      -> Membagi data Train (80%) & Test (20%) secara stratified
  5. cap_outliers()          -> Membatasi nilai ekstrem (Winsorizing) dari batas data Train
  6. encode_scale_and_save() -> One-Hot Encoding (kategori), StandardScaler (numerik), & simpan CSV
================================================================================
"""

import os
import re
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==============================================================================
# BAGIAN A: LOAD DATA & INITIAL INSPECTION (INSPEKSI AWAL DATA)
# ==============================================================================
# CATATAN (NOTES):
# - Tujuan bagian ini adalah mengecek kondisi awal data mentah sebelum diubah.
# - Hal yang dicek:
#   * Ukuran baris (jumlah mahasiswa) dan kolom (jumlah atribut).
#   * Apakah ada nilai kosong (Missing Values / NaN)?
#   * Apakah ada baris data yang kembar/duplikat?
#   * Bagaimana distribusi awal pada kolom 'target'?
# ==============================================================================
def load_and_inspect(input_path: str = "students_dropout_academic_success.csv") -> pd.DataFrame:
    print("=" * 70)
    print("[BAGIAN A] DATA LOADING & INITIAL INSPECTION")
    print("=" * 70)

    df_raw = pd.read_csv(input_path)

    print(f"Ukuran awal dataset : {df_raw.shape[0]} baris x {df_raw.shape[1]} kolom")
    print(f"Total missing values: {df_raw.isnull().sum().sum()}")
    print(f"Total baris duplikat: {df_raw.duplicated().sum()}")

    print("\nDistribusi Target Awal (3 Kelas):")
    print(df_raw["target"].value_counts())

    return df_raw


# ==============================================================================
# BAGIAN B: DATA CLEANING & TARGET FILTERING (PEMBERSIHAN DATA)
# ==============================================================================
# CATATAN (NOTES):
# - Apa saja yang dibersihkan di bagian ini?
#   1. Nama Kolom: Di data asli banyak spasi, tanda kurung, kutip (Mother's),
#      dan typo ('Nacionality'). Semua diubah ke snake_case (huruf kecil + underscore)
#      supaya konsisten dan mudah dipanggil di kode Python.
#   2. Hapus Duplikat: Memastikan tidak ada baris mahasiswa yang tercatat ganda.
#   3. Filter Target (Dropout vs Graduate):
#      - Di data asli ada 3 status: 'Graduate', 'Dropout', dan 'Enrolled'.
#      - Karena fokus kita memprediksi Dropout vs Graduate, baris 'Enrolled'
#        (masih kuliah) dibuang karena hasil akhirnya belum diketahui.
#      - Label teks diubah menjadi angka biner:
#        * Graduate -> 0
#        * Dropout  -> 1
#   4. Perbaikan Anomali Data:
#      - Kolom 'application_order' seharusnya bernilai 1 (pilihan pertama) s/d 9.
#      - Ada baris yang bernilai 0, sehingga dikoreksi menjadi 1.
# ==============================================================================
def clean_column_name(col: str) -> str:
    """Mengubah nama kolom menjadi snake_case yang bersih dan memperbaiki typo."""
    col = col.strip().lower()
    col = col.replace("nacionality", "nationality")  # Fix typo bawaan dataset
    col = col.replace("'", "")
    col = re.sub(r"[^a-z0-9]+", "_", col)
    return col.strip("_")


def clean_and_filter_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("[BAGIAN B] DATA CLEANING & FILTERING (DROPOUT VS GRADUATE)")
    print("=" * 70)

    df = df_raw.copy()

    # 1. Standarisasi nama kolom ke snake_case
    df.columns = [clean_column_name(c) for c in df.columns]
    print("Contoh nama kolom setelah dibersihkan:", list(df.columns[:6]))

    # 2. Hapus baris duplikat (jika ada)
    df = df.drop_duplicates().reset_index(drop=True)

    # 3. Filter khusus Binary Classification: Dropout (1) vs Graduate (0)
    df = df[df["target"].isin(["Dropout", "Graduate"])].copy().reset_index(drop=True)
    df["target"] = df["target"].map({"Graduate": 0, "Dropout": 1}).astype(int)

    grad_count = (df["target"] == 0).sum()
    drop_count = (df["target"] == 1).sum()
    print(f"Ukuran setelah filter (Dropout vs Graduate): {df.shape[0]} baris")
    print(
        f"Distribusi Target Baru -> Graduate (0): {grad_count} ({grad_count/len(df)*100:.1f}%), "
        f"Dropout (1): {drop_count} ({drop_count/len(df)*100:.1f}%)"
    )

    # 4. Perbaiki anomali nilai pada application_order (0 -> 1)
    anomaly_count = (df["application_order"] == 0).sum()
    if anomaly_count > 0:
        df["application_order"] = df["application_order"].replace(0, 1)
        print(f"Memperbaiki {anomaly_count} baris anomali pada 'application_order' (0 -> 1)")

    return df


# ==============================================================================
# BAGIAN C: FEATURE ENGINEERING (MEMBUAT FITUR BARU)
# ==============================================================================
# CATATAN (NOTES):
# - Mengapa perlu membuat fitur baru?
#   Jika model hanya tahu "lulus 4 matkul", model tidak tahu apakah mahasiswa
#   tersebut mengambil 4 matkul (lulus 100%) atau 10 matkul (lulus 40%).
# - Fitur turunan yang ditambahkan:
#   1. approval_rate_1st_sem : Rasio matkul lulus dibanding yang diambil di Sem 1.
#   2. approval_rate_2nd_sem : Rasio matkul lulus dibanding yang diambil di Sem 2.
#   3. grade_diff_sem2_sem1  : Selisih nilai rata-rata Sem 2 dikurangi Sem 1.
#   4. approval_rate_diff    : Selisih rasio kelulusan Sem 2 dikurangi Sem 1.
#   5. total_enrolled_units  : Total matkul yang diambil selama tahun pertama.
#   6. total_approved_units  : Total matkul yang lulus selama tahun pertama.
# ==============================================================================
def engineer_features(df_clean: pd.DataFrame, output_dir: str = "processed_data") -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("[BAGIAN C] FEATURE ENGINEERING (MEMBUAT FITUR BARU)")
    print("=" * 70)

    df = df_clean.copy()
    os.makedirs(output_dir, exist_ok=True)

    # 1. Rasio kelulusan mata kuliah per semester (Approved / Enrolled)
    # Jika enrolled == 0, diisi 0.0 agar tidak terjadi pembagian dengan nol
    df["approval_rate_1st_sem"] = np.where(
        df["curricular_units_1st_sem_enrolled"] > 0,
        df["curricular_units_1st_sem_approved"] / df["curricular_units_1st_sem_enrolled"],
        0.0,
    )
    df["approval_rate_2nd_sem"] = np.where(
        df["curricular_units_2nd_sem_enrolled"] > 0,
        df["curricular_units_2nd_sem_approved"] / df["curricular_units_2nd_sem_enrolled"],
        0.0,
    )

    # 2. Perubahan performa dari semester 1 ke semester 2
    df["grade_diff_sem2_sem1"] = (
        df["curricular_units_2nd_sem_grade"] - df["curricular_units_1st_sem_grade"]
    )
    df["approval_rate_diff"] = df["approval_rate_2nd_sem"] - df["approval_rate_1st_sem"]

    # 3. Total mata kuliah yang diambil dan lulus selama tahun pertama
    df["total_enrolled_units"] = (
        df["curricular_units_1st_sem_enrolled"] + df["curricular_units_2nd_sem_enrolled"]
    )
    df["total_approved_units"] = (
        df["curricular_units_1st_sem_approved"] + df["curricular_units_2nd_sem_approved"]
    )

    engineered_cols = [
        "approval_rate_1st_sem",
        "approval_rate_2nd_sem",
        "grade_diff_sem2_sem1",
        "approval_rate_diff",
        "total_enrolled_units",
        "total_approved_units",
    ]
    print(f"Berhasil menambahkan {len(engineered_cols)} fitur baru:")
    for col in engineered_cols:
        print(f"  + {col}")

    # Simpan dataset bersih (sebelum scaling/one-hot) untuk kebutuhan EDA/visualisasi
    clean_unscaled_path = os.path.join(output_dir, "data_cleaned_unscaled.csv")
    df.to_csv(clean_unscaled_path, index=False)
    print(f"\n[INFO] Dataset bersih (unscaled) disimpan ke: {clean_unscaled_path}")

    return df


# ==============================================================================
# BAGIAN D: TRAIN-TEST SPLIT (MEMBAGI DATA LATIH & DATA UJI)
# ==============================================================================
# CATATAN PENTING (NOTES):
# - Train-Test Split wajib dilakukan SEBELUM Outlier Capping, Imputasi, dan Scaling
#   agar TIDAK terjadi DATA LEAKAGE (kebocoran informasi dari data uji ke data latih).
# - Semua batas persentil outlier, nilai median, dan rata-rata scaling nantinya
#   dihitung dari X_train saja, lalu diterapkan ke X_test.
# - Parameter `stratify=y` menjaga proporsi kelas Dropout (39.1%) dan Graduate (60.9%)
#   tetap seimbang di data Train maupun Test.
# ==============================================================================
def split_train_test(
    df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    print("\n" + "=" * 70)
    print("[BAGIAN D] TRAIN-TEST SPLIT (MENCEGAH DATA LEAKAGE)")
    print("=" * 70)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Ukuran Data Latih (Train): X_train = {X_train.shape}, y_train = {y_train.shape}")
    print(f"Ukuran Data Uji   (Test) : X_test  = {X_test.shape},  y_test  = {y_test.shape}")
    print(f"Proporsi Dropout di Train: {y_train.mean()*100:.2f}%")
    print(f"Proporsi Dropout di Test : {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test


# ==============================================================================
# BAGIAN E: PENANGANAN OUTLIER (WINSORIZING / CAPPING)
# ==============================================================================
# CATATAN (NOTES):
# - Outlier adalah nilai yang terlalu ekstrem dibanding mayoritas data
#   (contoh: umur mahasiswa saat mendaftar 70 tahun).
# - Alih-alih menghapus barisnya, kita gunakan Capping (Winsorizing) pada
#   persentil ke-1 (1%) dan ke-99 (99%) agar efek ekstremnya teredam tanpa
#   kehilangan sampel data.
# - Batas bawah (1%) dan batas atas (99%) dihitung HANYA dari X_train!
# ==============================================================================
def cap_outliers(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 70)
    print("[BAGIAN E] PENANGANAN OUTLIER (CAPPING PADA DATA TRAIN)")
    print("=" * 70)

    outlier_cols = [
        "age_at_enrollment",
        "admission_grade",
        "previous_qualification_grade",
    ]

    X_train_capped = X_train.copy()
    X_test_capped = X_test.copy()

    for col in outlier_cols:
        lower_bound = X_train_capped[col].quantile(0.01)
        upper_bound = X_train_capped[col].quantile(0.99)

        X_train_capped[col] = X_train_capped[col].clip(lower=lower_bound, upper=upper_bound)
        X_test_capped[col] = X_test_capped[col].clip(lower=lower_bound, upper=upper_bound)

        print(
            f"- Kolom '{col}' dibatasi pada rentang [{lower_bound:.2f} s/d {upper_bound:.2f}]"
        )

    return X_train_capped, X_test_capped


# ==============================================================================
# BAGIAN F: ENCODING KATEGORI, FEATURE SCALING & EXPORT DATA
# ==============================================================================
# CATATAN PENTING (NOTES):
# Walaupun semua kolom di CSV terlihat berisi angka, sifat aslinya berbeda-beda
# sehingga dibagi menjadi 3 kelompok perlakuan:
#
# 1. KOLOM BINER (8 Kolom):
#    - Contoh: gender, debtor, scholarship_holder, tuition_fees_up_to_date.
#    - Nilainya sudah 0 atau 1, tidak perlu diubah lagi.
#
# 2. KOLOM KATEGORI NOMINAL (9 Kolom):
#    - Contoh: course (kode jurusan 33 s/d 9991), marital_status, application_mode,
#      nationality, mothers_occupation, fathers_occupation.
#    - Angka di sini hanyalah KODE ID kategori, bukan urutan besaran!
#    - Diubah menjadi kolom biner (0/1) dengan `OneHotEncoder(min_frequency=0.01)`
#      agar kategori langka (< 1% data) digabung otomatis ke kolom 'infrequent'.
#
# 3. KOLOM NUMERIK & ORDINAL (25 Kolom):
#    - Contoh: admission_grade, gdp, jumlah SKS, umur, dan 6 fitur baru.
#    - Disamakan skalanya menggunakan `StandardScaler` (rata-rata = 0, std = 1).
# ==============================================================================
def encode_scale_and_save(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str = "processed_data",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 70)
    print("[BAGIAN F] ENCODING KATEGORI, FEATURE SCALING & SIMPAN DATA")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Daftar kolom Biner (0 atau 1)
    binary_cols = [
        "daytime_evening_attendance",
        "displaced",
        "educational_special_needs",
        "debtor",
        "tuition_fees_up_to_date",
        "gender",
        "scholarship_holder",
        "international",
    ]

    # 2. Daftar kolom Kategori Nominal (Kode ID angka -> One-Hot Encoding)
    nominal_categorical_cols = [
        "marital_status",
        "application_mode",
        "course",
        "previous_qualification",
        "nationality",
        "mothers_qualification",
        "fathers_qualification",
        "mothers_occupation",
        "fathers_occupation",
    ]

    # 3. Daftar kolom Numerik & Ordinal (Sisanya -> StandardScaler)
    numeric_cols = [
        col
        for col in X_train.columns
        if col not in binary_cols and col not in nominal_categorical_cols
    ]

    print(f"- Jumlah fitur Numerik/Ordinal : {len(numeric_cols)} kolom -> StandardScaler")
    print(f"- Jumlah fitur Kategori Nominal: {len(nominal_categorical_cols)} kolom  -> OneHotEncoder")
    print(f"- Jumlah fitur Biner (0/1)     : {len(binary_cols)} kolom  -> Passthrough (Tetap 0/1)")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=0.01,
                    sparse_output=False,
                ),
            ),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, nominal_categorical_cols),
            ("bin", binary_pipeline, binary_cols),
        ],
        verbose_feature_names_out=False,
    )

    # Fit HANYA pada X_train, lalu transform X_train dan X_test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    print(f"\nDimensi akhir setelah One-Hot Encoding & Scaling:")
    print(f"  * X_train_df: {X_train_df.shape[0]} baris x {X_train_df.shape[1]} kolom")
    print(f"  * X_test_df : {X_test_df.shape[0]} baris x {X_test_df.shape[1]} kolom")

    # Simpan semua hasil akhir ke folder processed_data/
    X_train_df.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test_df.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))

    print("\n" + "=" * 70)
    print(f"SELESAI! Semua file tersimpan di folder '{output_dir}/':")
    print("  1. data_cleaned_unscaled.csv (Data bersih sebelum scaling/one-hot)")
    print("  2. X_train.csv & y_train.csv (Data latih siap masuk model ML)")
    print("  3. X_test.csv  & y_test.csv  (Data uji siap evaluasi model ML)")
    print("  4. preprocessor.joblib       (Pipeline scikit-learn untuk deployment)")
    print("=" * 70)

    return X_train_df, X_test_df


# ==============================================================================
# EKSEKUSI UTAMA (MAIN PIPELINE)
# ==============================================================================
def main():
    df_raw = load_and_inspect("students_dropout_academic_success.csv")
    df_clean = clean_and_filter_data(df_raw)
    df_featured = engineer_features(df_clean, output_dir="processed_data")
    X_train, X_test, y_train, y_test = split_train_test(
        df_featured, test_size=0.20, random_state=42
    )
    X_train_capped, X_test_capped = cap_outliers(X_train, X_test)
    encode_scale_and_save(
        X_train_capped,
        X_test_capped,
        y_train,
        y_test,
        output_dir="processed_data",
    )


if __name__ == "__main__":
    main()
