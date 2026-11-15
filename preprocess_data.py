"""
================================================================================
SISTEM PRESKRIPTIF DETEKSI DINI RISIKO DROP-OUT MAHASISWA
MENGGUNAKAN RANDOM FOREST DAN SHAP EXPLAINABILITY
--------------------------------------------------------------------------------
TAHAP 1 & TAHAP 2: EKSPLORASI DATA (EDA), PEMISAHAN DOMAIN FITUR,
DAN PREPROCESSING ANTI-DATA LEAKAGE (ONE-HOT ENCODING + TRAIN/TEST SPLIT + SMOTE)
================================================================================
Dataset : Students' Dropout and Academic Success (UCI ML Repository)
Target  : Target_Binary -> Dropout (1) vs Graduate (0) [Baris 'Enrolled' di-drop]

Prinsip Anti-Data Leakage:
1. Pemisahan Domain Fitur (Demografi, Sosial-Ekonomi, Admisi, Akademik Sem 1 & Sem 2).
2. Train-Test Split (Stratified) dilakukan TERLEBIH DAHULU sebelum transformasi/SMOTE.
3. One-Hot Encoding & Scaling di-fit HANYA pada X_train, lalu di-transform ke X_test.
4. Algoritma SMOTE diterapkan HANYA pada X_train_encoded & y_train.
   Test set (X_test_encoded, y_test) dibiarkan murni tanpa data sintetis.
================================================================================
"""

import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Coba gunakan SMOTE dari imbalanced-learn jika tersedia,
# atau gunakan implementasi identik berbasis scikit-learn NearestNeighbors.
try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

RANDOM_STATE = 42


class SklearnSMOTE:
    """
    Implementasi algoritma SMOTE (Chawla et al., 2002) berbasis scikit-learn
    NearestNeighbors sebagai fallback jika paket imbalanced-learn belum terinstal.
    """

    def __init__(self, k_neighbors: int = 5, random_state: int = RANDOM_STATE):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        rng = np.random.default_rng(self.random_state)
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        classes, counts = np.unique(y_arr, return_counts=True)
        maj_count = counts.max()

        X_resampled_list = [X_arr]
        y_resampled_list = [y_arr]

        for cls, count in zip(classes, counts):
            n_samples_needed = maj_count - count
            if n_samples_needed <= 0:
                continue

            X_min = X_arr[y_arr == cls]
            nn = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
            nn.fit(X_min)
            neighbors = nn.kneighbors(X_min, return_distance=False)[:, 1:]

            sample_indices = rng.integers(0, len(X_min), size=n_samples_needed)
            neighbor_SubIndices = rng.integers(0, self.k_neighbors, size=n_samples_needed)
            neighbor_indices = neighbors[sample_indices, neighbor_SubIndices]

            steps = rng.random(size=(n_samples_needed, 1))
            synthetic_X = X_min[sample_indices] + steps * (
                X_min[neighbor_indices] - X_min[sample_indices]
            )
            synthetic_y = np.full(n_samples_needed, cls, dtype=y_arr.dtype)

            X_resampled_list.append(synthetic_X)
            y_resampled_list.append(synthetic_y)

        X_out = np.vstack(X_resampled_list)
        y_out = np.concatenate(y_resampled_list)

        if isinstance(X, pd.DataFrame):
            X_out = pd.DataFrame(X_out, columns=X.columns)
        if isinstance(y, pd.Series):
            y_out = pd.Series(y_out, name=y.name)

        return X_out, y_out


# ==============================================================================
# TAHAP 1: EKSPLORASI DAN PEMISAHAN DOMAIN FITUR (EDA)
# ==============================================================================
def load_and_prepare_eda(
    filepath: str = "students_dropout_academic_success.csv",
    include_semester_2: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    print("=" * 75)
    print("TAHAP 1: EKSPLORASI DATA & PEMISAHAN DOMAIN FITUR (EDA)")
    print("=" * 75)

    if not os.path.exists(filepath):
        if os.path.exists("data/data.csv"):
            filepath = "data/data.csv"
        else:
            raise FileNotFoundError(f"Dataset tidak ditemukan pada path: {filepath}")

    # 1. Load Data (Deteksi otomatis delimiter ';' atau ',')
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline()
    sep = ";" if ";" in first_line else ","
    df = pd.read_csv(filepath, sep=sep)

    # Normalisasi nama kolom minor agar 100% sesuai standar dokumentasi UCI
    col_rename_map = {
        "Marital Status": "Marital status",
        "Daytime/evening attendance\t": "Daytime/evening attendance",
        "target": "Target",
    }
    df = df.rename(columns=col_rename_map)

    print(f"Total Data Awal (Sebelum Filter): {df.shape[0]} baris x {df.shape[1]} kolom")
    print("Distribusi Target Awal:")
    for label, count in df["Target"].value_counts().items():
        print(f"  - {label:<10}: {count} ({count / len(df) * 100:.2f}%)")

    # 2. Hapus status 'Enrolled' untuk fokus pada Binary Classification tegas
    df = df[df["Target"] != "Enrolled"].copy().reset_index(drop=True)

    # 3. Label Encoding untuk Target: Dropout = 1, Graduate = 0
    df["Target_Binary"] = df["Target"].map({"Dropout": 1, "Graduate": 0})
    df = df.drop(columns=["Target"])

    # 4. Definisi List Pemisahan Domain Fitur (36 Fitur Independen)
    features_demography = [
        "Marital status",
        "Nacionality",
        "Displaced",
        "Gender",
        "Age at enrollment",
        "International",
    ]

    features_socioeconomic = [
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
        "Educational special needs",
        "Debtor",
        "Tuition fees up to date",
        "Scholarship holder",
        "Unemployment rate",
        "Inflation rate",
        "GDP",
    ]

    features_admission = [
        "Application mode",
        "Application order",
        "Course",
        "Daytime/evening attendance",
        "Previous qualification",
        "Previous qualification (grade)",
        "Admission grade",
    ]

    features_academic_s1 = [col for col in df.columns if "1st sem" in col]
    features_academic_s2 = [col for col in df.columns if "2nd sem" in col]

    # Jika skenario deteksi dini di akhir Semester 1, drop fitur Semester 2
    if not include_semester_2:
        df = df.drop(columns=features_academic_s2)
        print("\n[INFO] Mode Deteksi Dini Semester 1 aktif: Fitur Semester 2 di-drop.")

    feature_domains = {
        "demography": features_demography,
        "socioeconomic": features_socioeconomic,
        "admission": features_admission,
        "academic_s1": features_academic_s1,
        "academic_s2": features_academic_s2 if include_semester_2 else [],
    }

    # 5. Sanity Check Eksekusi
    print(f"\nTotal Data setelah drop 'Enrolled': {df.shape}")
    print("Distribusi Target_Binary (Imbalance Check):")
    dist = df["Target_Binary"].value_counts(normalize=True) * 100
    counts = df["Target_Binary"].value_counts()
    print(f"  - Graduate (0): {counts[0]} mahasiswa ({dist[0]:.2f}%)")
    print(f"  - Dropout  (1): {counts[1]} mahasiswa ({dist[1]:.2f}%)")

    # Cek apakah ada missing values
    missing_total = df.isnull().sum().sum()
    assert missing_total == 0, "Ditemukan missing values, butuh imputasi!"
    print(f"Cek Missing Values: {missing_total} (Aman, 0 missing values)")

    print("\nRingkasan Pemisahan Domain Fitur:")
    total_f = 0
    for domain, cols in feature_domains.items():
        print(f"  - {domain:<15}: {len(cols)} fitur")
        total_f += len(cols)
    print(f"  Total Fitur Aktif : {total_f} fitur")

    return df, feature_domains


# ==============================================================================
# TAHAP 2: PREPROCESSING, ONE-HOT ENCODING, TRAIN-TEST SPLIT & SMOTE
# ==============================================================================
def preprocess_and_split_no_leakage(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
    output_dir: str = "processed_data",
) -> None:
    print("\n" + "=" * 75)
    print("TAHAP 2: PREPROCESSING & PEMISAHAN DATA (MENCEGAH DATA LEAKAGE)")
    print("=" * 75)

    target_col = "Target_Binary"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # --------------------------------------------------------------------------
    # 1. Identifikasi Tipe Fitur: Kategorikal Nominal vs Biner vs Numerik
    # --------------------------------------------------------------------------
    # Fitur Kategorikal Nominal (kode angka di UCI yang merepresentasikan kategori/jurusan/status)
    categorical_nominal_cols = [
        "Marital status",
        "Application mode",
        "Course",
        "Previous qualification",
        "Nacionality",
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
    ]
    categorical_nominal_cols = [c for c in categorical_nominal_cols if c in X.columns]

    # Fitur Biner (sudah 0 atau 1, tidak perlu di-One-Hot Encode ulang)
    binary_cols = [
        "Daytime/evening attendance",
        "Displaced",
        "Educational special needs",
        "Debtor",
        "Tuition fees up to date",
        "Gender",
        "Scholarship holder",
        "International",
    ]
    binary_cols = [c for c in binary_cols if c in X.columns]

    # Fitur Numerik (sisanya: nilai, usia, SKS, makroekonomi)
    numeric_cols = [
        c for c in X.columns if c not in categorical_nominal_cols + binary_cols
    ]

    print(f"Klasifikasi Tipe Kolom ({X.shape[1]} Fitur):")
    print(f"  - Kategorikal Nominal (One-Hot Encoding) : {len(categorical_nominal_cols)} fitur")
    print(f"  - Biner (0/1 Passthrough)                : {len(binary_cols)} fitur")
    print(f"  - Numerik (StandardScaler)               : {len(numeric_cols)} fitur")

    # --------------------------------------------------------------------------
    # 2. TRAIN-TEST SPLIT TERLEBIH DAHULU (SEBELUM ENCODING & SMOTE!)
    # --------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("\n[1] Hasil Stratified Train-Test Split (Sebelum Encoding & Balancing):")
    print(f"  - X_train : {X_train.shape} | Distribusi y_train -> Graduate(0): {(y_train == 0).sum()}, Dropout(1): {(y_train == 1).sum()}")
    print(f"  - X_test  : {X_test.shape}  | Distribusi y_test  -> Graduate(0): {(y_test == 0).sum()}, Dropout(1): {(y_test == 1).sum()}")

    # --------------------------------------------------------------------------
    # 3. ONE-HOT ENCODING & SCALING (FIT HANYA PADA X_TRAIN)
    # --------------------------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("bin", "passthrough", binary_cols),
            (
                "cat",
                OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                categorical_nominal_cols,
            ),
        ],
        verbose_feature_names_out=False,
    )

    # Fit HANYA pada X_train, lalu transform X_train dan X_test
    X_train_encoded_arr = preprocessor.fit_transform(X_train)
    X_test_encoded_arr = preprocessor.transform(X_test)

    feature_names_out = list(preprocessor.get_feature_names_out())
    X_train_encoded = pd.DataFrame(X_train_encoded_arr, columns=feature_names_out)
    X_test_encoded = pd.DataFrame(X_test_encoded_arr, columns=feature_names_out)
    y_train_reset = y_train.reset_index(drop=True)
    y_test_reset = y_test.reset_index(drop=True)

    print("\n[2] Hasil One-Hot Encoding & Scaling (Fit hanya pada X_train):")
    print(f"  - Dimensi X_train_encoded : {X_train_encoded.shape}")
    print(f"  - Dimensi X_test_encoded  : {X_test_encoded.shape}")

    # --------------------------------------------------------------------------
    # 4. BALANCING DENGAN SMOTE (HANYA PADA DATA TRAINING!)
    # --------------------------------------------------------------------------
    if HAS_IMBLEARN:
        smote = SMOTE(random_state=random_state)
        smote_engine_name = "imblearn.over_sampling.SMOTE"
    else:
        smote = SklearnSMOTE(k_neighbors=5, random_state=random_state)
        smote_engine_name = "SklearnSMOTE (Chawla et al. k=5)"

    X_train_smote, y_train_smote = smote.fit_resample(X_train_encoded, y_train_reset)

    print(f"\n[3] Hasil Balancing dengan SMOTE ({smote_engine_name}) HANYA pada Train Set:")
    print(f"  - Sebelum SMOTE (y_train)       : Graduate(0) = {(y_train_reset == 0).sum()}, Dropout(1) = {(y_train_reset == 1).sum()} (Total: {len(y_train_reset)})")
    print(f"  - Sesudah SMOTE (y_train_smote) : Graduate(0) = {(y_train_smote == 0).sum()}, Dropout(1) = {(y_train_smote == 1).sum()} (Total: {len(y_train_smote)})")
    print(f"  - Test Set (y_test - MURNI)     : Graduate(0) = {(y_test_reset == 0).sum()}, Dropout(1) = {(y_test_reset == 1).sum()} (Total: {len(y_test_reset)})")

    # --------------------------------------------------------------------------
    # 5. SIMPAN ARTIFAK DATA & PREPROCESSOR
    # --------------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # Simpan data yang sudah di-encode & di-SMOTE (untuk pelatihan model)
    X_train_smote.to_csv(os.path.join(output_dir, "X_train_smote.csv"), index=False)
    y_train_smote.to_frame(name=target_col).to_csv(os.path.join(output_dir, "y_train_smote.csv"), index=False)

    # Simpan juga X_train sebelum SMOTE (untuk pembanding eksperimen tanpa SMOTE / analisis XAI)
    X_train_encoded.to_csv(os.path.join(output_dir, "X_train_encoded.csv"), index=False)
    y_train_reset.to_frame(name=target_col).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)

    # Simpan Test Set murni (tidak tersentuh SMOTE)
    X_test_encoded.to_csv(os.path.join(output_dir, "X_test_encoded.csv"), index=False)
    y_test_reset.to_frame(name=target_col).to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    # Simpan X_train & X_test sebelum OHE (untuk analisis EDA / interpretasi domain asli)
    X_train.reset_index(drop=True).to_csv(os.path.join(output_dir, "X_train_raw.csv"), index=False)
    X_test.reset_index(drop=True).to_csv(os.path.join(output_dir, "X_test_raw.csv"), index=False)

    # Simpan objek preprocessor
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))

    print(f"\n[4] Seluruh file berhasil disimpan di folder '{output_dir}/':")
    print("  - X_train_smote.csv & y_train_smote.csv (Data Latih Seimbang 50:50)")
    print("  - X_train_encoded.csv & y_train.csv     (Data Latih Asli Sebelum SMOTE)")
    print("  - X_test_encoded.csv & y_test.csv       (Data Uji Murni Bebas Leakage)")
    print("  - X_train_raw.csv & X_test_raw.csv      (Data Sebelum One-Hot Encoding)")
    print("  - preprocessor.joblib                   (Pipeline ColumnTransformer)")
    print("=" * 75)


if __name__ == "__main__":
    df_clean, domains = load_and_prepare_eda(
        filepath="students_dropout_academic_success.csv",
        include_semester_2=True,
    )
    preprocess_and_split_no_leakage(df_clean)
