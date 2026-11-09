"""
================================================================================
PIPELINE DATA PREPROCESSING — PREDIKSI PERFORMA AKADEMIK MAHASISWA (P08)
PENDEKATAN REGRESI NILAI AKHIR (CURRICULAR_UNITS_2ND_SEM_GRADE)
================================================================================
Modul ini disusun oleh Data Engineer untuk memenuhi Bab 2 & Bab 3 Panduan
Pengerjaan Proyek P08. Seluruh alur dirancang bebas kebocoran target (target leakage),
menyerahkan data bersih dalam format 31 fitur asli tanpa normalisasi / One-Hot Encoding
(transformasi model diserahkan sepenuhnya ke Pipeline ML Engineer).

Tahapan Fungsi:
  1. load_and_inspect()      -> Membaca CSV mentah & inspeksi target regresi
  2. clean_and_filter_data() -> Snake_case, koreksi anomali, & eliminasi leakage Sem 2
  3. engineer_features()     -> Rekayasa fitur aman (approval_rate_1st_sem)
  4. split_train_test()      -> Split 80% Train / 20% Test (random_state=42)
  5. cap_outliers()          -> Winsorizing persentil 1% & 99% dari X_train
  6. save_processed_data()   -> Validasi tipe data & simpan berkas bersih ke processed_data/
================================================================================
"""

import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# Definisi Kontrak 31 Fitur Input (Sesuai DATA_DICTIONARY.md & PROJECT_NOTES_P08.md)
BINARY_COLS = [
    "daytime_evening_attendance",
    "displaced",
    "educational_special_needs",
    "debtor",
    "tuition_fees_up_to_date",
    "gender",
    "scholarship_holder",
    "international",
]

NOMINAL_CATEGORICAL_COLS = [
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

NUMERIC_COLS = [
    "application_order",
    "previous_qualification_grade",
    "admission_grade",
    "age_at_enrollment",
    "curricular_units_1st_sem_credited",
    "curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_evaluations",
    "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_grade",
    "curricular_units_1st_sem_without_evaluations",
    "unemployment_rate",
    "inflation_rate",
    "gdp",
    "approval_rate_1st_sem",
]

TARGET_COL = "curricular_units_2nd_sem_grade"

LEAKAGE_COLS_TO_DROP = [
    "target",
    "curricular_units_2nd_sem_credited",
    "curricular_units_2nd_sem_enrolled",
    "curricular_units_2nd_sem_evaluations",
    "curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_without_evaluations",
]


# ==============================================================================
# BAGIAN A: LOAD DATA & INITIAL INSPECTION (INSPEKSI AWAL DATA)
# ==============================================================================
def load_and_inspect(input_path: str = "students_dropout_academic_success.csv") -> pd.DataFrame:
    print("=" * 70)
    print("[BAGIAN A] DATA LOADING & INITIAL INSPECTION")
    print("=" * 70)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Berkas dataset tidak ditemukan di: {input_path}")

    df_raw = pd.read_csv(input_path)

    print(f"Ukuran awal dataset  : {df_raw.shape[0]} baris x {df_raw.shape[1]} kolom")
    print(f"Total missing values : {df_raw.isnull().sum().sum()}")
    print(f"Total baris duplikat : {df_raw.duplicated().sum()}")

    # Evaluasi awal target regresi (curricular units 2nd sem (grade))
    # Mencari nama kolom target di dataset mentah sebelum pembersihan nama
    raw_target_col = [c for c in df_raw.columns if "2nd sem" in c.lower() and "grade" in c.lower()][0]
    target_series = df_raw[raw_target_col]

    zero_count = (target_series == 0.0).sum()
    print(f"\nStatistik Target Regresi Mentah ('{raw_target_col}'):")
    print(f"  - Mean   : {target_series.mean():.2f}")
    print(f"  - Std    : {target_series.std():.2f}")
    print(f"  - Min    : {target_series.min():.2f}")
    print(f"  - Median : {target_series.median():.2f}")
    print(f"  - Max    : {target_series.max():.2f}")
    print(f"  - Nilai 0.0 (Zero-Inflated) : {zero_count} mahasiswa ({zero_count/len(df_raw)*100:.2f}%)")

    return df_raw


# ==============================================================================
# BAGIAN B: DATA CLEANING & ELIMINASI TARGET LEAKAGE
# ==============================================================================
def clean_column_name(col: str) -> str:
    """Mengubah nama kolom menjadi snake_case yang bersih dan memperbaiki typo."""
    col = col.strip().lower()
    col = col.replace("nacionality", "nationality")  # Fix typo bawaan dataset UCI
    col = col.replace("'", "")
    col = re.sub(r"[^a-z0-9]+", "_", col)
    return col.strip("_")


def clean_and_filter_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("[BAGIAN B] DATA CLEANING & ELIMINASI TARGET LEAKAGE")
    print("=" * 70)

    df = df_raw.copy()

    # 1. Standarisasi nama kolom ke snake_case
    df.columns = [clean_column_name(c) for c in df.columns]

    # 2. Hapus duplikat baris (jika ada)
    df = df.drop_duplicates().reset_index(drop=True)

    # 3. Koreksi anomali application_order (nilai 0 diubah menjadi pilihan ke-1)
    anomaly_count = (df["application_order"] == 0).sum()
    if anomaly_count > 0:
        df["application_order"] = df["application_order"].replace(0, 1)
        print(f"Koreksi anomali: {anomaly_count} baris pada 'application_order' diubah dari 0 menjadi 1.")

    # 4. Eliminasi Kolom Target Leakage
    # Target utama adalah curricular_units_2nd_sem_grade. Kolom semester 2 lainnya dan kolom 'target'
    # kelulusan akhir WAJIB dibuang agar model bisa dipakai memprediksi di awal semester 2.
    existing_leakage = [col for col in LEAKAGE_COLS_TO_DROP if col in df.columns]
    df = df.drop(columns=existing_leakage)
    print(f"Berhasil mengeliminasi {len(existing_leakage)} kolom penyebab target leakage:")
    for col in existing_leakage:
        print(f"  - {col}")

    # Catatan: Kita TIDAK membuang baris 'Enrolled', seluruh 4.424 mahasiswa dipertahankan
    # karena target kita adalah performa nilai akademik seluruh mahasiswa yang berkuliah.
    print(f"Jumlah sampel dipertahankan: {df.shape[0]} baris (seluruh populasi mahasiswa).")

    return df


# ==============================================================================
# BAGIAN C: FEATURE ENGINEERING (REKAYASA FITUR BEBAS LEAKAGE)
# ==============================================================================
def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("[BAGIAN C] FEATURE ENGINEERING (BEBAS LEAKAGE)")
    print("=" * 70)

    df = df_clean.copy()

    # 1. Rasio kelulusan Semester 1: approval_rate_1st_sem = approved / enrolled
    # Jika enrolled == 0, diisi 0.0 agar aman dari pembagian nol.
    df["approval_rate_1st_sem"] = np.where(
        df["curricular_units_1st_sem_enrolled"] > 0,
        df["curricular_units_1st_sem_approved"] / df["curricular_units_1st_sem_enrolled"],
        0.0,
    )
    print("Fitur rekayasa ditambahkan: 'approval_rate_1st_sem' (rasio lulus SKS Sem 1).")

    # Catatan: Fitur rekayasa Semester 2 lama (approval_rate_2nd_sem, grade_diff_sem2_sem1,
    # approval_rate_diff, total_enrolled_units, total_approved_units) DITIADAKAN karena
    # membocorkan nilai Semester 2 secara langsung (100% target leakage).

    return df


# ==============================================================================
# BAGIAN D: TRAIN-TEST SPLIT (MENCEGAH DATA LEAKAGE)
# ==============================================================================
def split_train_test(
    df: pd.DataFrame, test_size: float = 0.20, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    print("\n" + "=" * 70)
    print("[BAGIAN D] TRAIN-TEST SPLIT (80% TRAIN / 20% TEST)")
    print("=" * 70)

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Menggunakan split tanpa stratify karena y adalah variabel kontinu regresi
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Ukuran Data Latih (Train): X_train = {X_train.shape}, y_train = {y_train.shape}")
    print(f"Ukuran Data Uji   (Test) : X_test  = {X_test.shape},  y_test  = {y_test.shape}")
    print(f"Target y_train mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")
    print(f"Target y_test  mean: {y_test.mean():.2f}, std: {y_test.std():.2f}")

    return X_train, X_test, y_train, y_test


# ==============================================================================
# BAGIAN E: PENANGANAN OUTLIER (WINSORIZING / CAPPING PADA X_TRAIN)
# ==============================================================================
def cap_outliers(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 70)
    print("[BAGIAN E] PENANGANAN OUTLIER (CAPPING HANYA DARI X_TRAIN)")
    print("=" * 70)

    outlier_cols = [
        "age_at_enrollment",
        "admission_grade",
        "previous_qualification_grade",
    ]

    X_train_capped = X_train.copy()
    X_test_capped = X_test.copy()

    for col in outlier_cols:
        lower_bound = float(X_train_capped[col].quantile(0.01))
        upper_bound = float(X_train_capped[col].quantile(0.99))

        X_train_capped[col] = X_train_capped[col].clip(lower=lower_bound, upper=upper_bound)
        X_test_capped[col] = X_test_capped[col].clip(lower=lower_bound, upper=upper_bound)

        print(f"- Kolom '{col}': dibatasi pada rentang [{lower_bound:.2f} s/d {upper_bound:.2f}]")

    return X_train_capped, X_test_capped


# ==============================================================================
# BAGIAN F: VALIDASI TIPE DATA & EKSPOR DATA BERSIH (TANPA SCALING / OHE)
# ==============================================================================
def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    df_clean: pd.DataFrame,
    output_dir: str = "processed_data",
) -> None:
    print("\n" + "=" * 70)
    print("[BAGIAN F] VALIDASI KONTRAK FITUR & PENYIMPANAN DATA BERSIH")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Pastikan urutan 31 kolom persis sesuai kontrak
    contract_cols = NUMERIC_COLS + NOMINAL_CATEGORICAL_COLS + BINARY_COLS
    assert len(contract_cols) == 31, f"Jumlah kolom kontrak harus 31, didapat {len(contract_cols)}"

    X_train_final = X_train[contract_cols].copy()
    X_test_final = X_test[contract_cols].copy()

    # 2. Pastikan tipe data konsisten
    for col in BINARY_COLS + NOMINAL_CATEGORICAL_COLS:
        X_train_final[col] = X_train_final[col].astype(int)
        X_test_final[col] = X_test_final[col].astype(int)

    for col in NUMERIC_COLS:
        if col in ["application_order", "age_at_enrollment", "curricular_units_1st_sem_credited",
                    "curricular_units_1st_sem_enrolled", "curricular_units_1st_sem_evaluations",
                    "curricular_units_1st_sem_approved", "curricular_units_1st_sem_without_evaluations"]:
            X_train_final[col] = X_train_final[col].astype(int)
            X_test_final[col] = X_test_final[col].astype(int)
        else:
            X_train_final[col] = X_train_final[col].astype(float)
            X_test_final[col] = X_test_final[col].astype(float)

    # 3. Hapus preprocessor.joblib lama agar tidak menimbulkan kebingungan arsitektur
    old_preprocessor = os.path.join(output_dir, "preprocessor.joblib")
    if os.path.exists(old_preprocessor):
        os.remove(old_preprocessor)
        print("Menghapus 'preprocessor.joblib' usang dari tahap Data Engineer.")

    # 4. Simpan berkas bersih ke processed_data/
    # Simpan dataset lengkap unscaled (31 fitur + target = 32 kolom) untuk EDA
    clean_unscaled_path = os.path.join(output_dir, "data_cleaned_unscaled.csv")
    df_clean.to_csv(clean_unscaled_path, index=False)

    # Simpan split Train dan Test
    X_train_path = os.path.join(output_dir, "X_train.csv")
    X_test_path = os.path.join(output_dir, "X_test.csv")
    y_train_path = os.path.join(output_dir, "y_train.csv")
    y_test_path = os.path.join(output_dir, "y_test.csv")

    X_train_final.to_csv(X_train_path, index=False)
    X_test_final.to_csv(X_test_path, index=False)
    y_train.to_frame(name=TARGET_COL).to_csv(y_train_path, index=False)
    y_test.to_frame(name=TARGET_COL).to_csv(y_test_path, index=False)

    # Juga sediakan train.csv dan test.csv gabungan (fitur + target) sesuai panduan Bab 2
    train_combined = pd.concat([X_train_final, y_train.to_frame(name=TARGET_COL)], axis=1)
    test_combined = pd.concat([X_test_final, y_test.to_frame(name=TARGET_COL)], axis=1)
    train_combined.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_combined.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print("Berhasil menyimpan berkas bersih ke folder 'processed_data/':")
    print(f"  1. X_train.csv ({X_train_final.shape[0]} baris x {X_train_final.shape[1]} kolom)")
    print(f"  2. X_test.csv  ({X_test_final.shape[0]} baris x {X_test_final.shape[1]} kolom)")
    print(f"  3. y_train.csv ({y_train.shape[0]} baris x 1 kolom target)")
    print(f"  4. y_test.csv  ({y_test.shape[0]} baris x 1 kolom target)")
    print(f"  5. train.csv & test.csv (format gabungan fitur + target)")
    print(f"  6. data_cleaned_unscaled.csv ({df_clean.shape[0]} baris x {df_clean.shape[1]} kolom)")
    print("=" * 70)


# ==============================================================================
# EKSEKUSI UTAMA (MAIN PIPELINE)
# ==============================================================================
def main():
    df_raw = load_and_inspect("students_dropout_academic_success.csv")
    df_clean = clean_and_filter_data(df_raw)
    df_featured = engineer_features(df_clean)
    X_train, X_test, y_train, y_test = split_train_test(
        df_featured, test_size=0.20, random_state=RANDOM_STATE
    )
    X_train_capped, X_test_capped = cap_outliers(X_train, X_test)
    save_processed_data(
        X_train_capped,
        X_test_capped,
        y_train,
        y_test,
        df_clean=df_featured,
        output_dir="processed_data",
    )


if __name__ == "__main__":
    main()
