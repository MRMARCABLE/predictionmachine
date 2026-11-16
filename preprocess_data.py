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

Struktur Output Folder `processed_data/`:
  ├── 1_raw_split/     -> Data hasil split 80:20 sebelum di-encode (36 fitur asli)
  ├── 2_encoded/       -> Data hasil One-Hot Encoding & Scaling (227 fitur) + preprocessor
  └── 3_train_smote/   -> Data Training hasil balancing SMOTE 50:50 (siap masuk model)
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

RANDOM_STATE = 42


class SklearnSMOTE:
    """
    Implementasi algoritma SMOTE (Chawla et al., 2002) berbasis scikit-learn
    NearestNeighbors (k=5) untuk menyeimbangkan kelas minoritas (Dropout)
    hanya pada data training.
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
            neighbor_sub_indices = rng.integers(0, self.k_neighbors, size=n_samples_needed)
            neighbor_indices = neighbors[sample_indices, neighbor_sub_indices]

            steps = rng.random(size=(n_samples_needed, 1))
            synthetic_X = X_min[sample_indices] + steps * (
                X_min[neighbor_indices] - X_min[sample_indices]
            )
            synthetic_y = np.full(n_samples_needed, cls, dtype=y_arr.dtype)

            X_resampled_list.append(synthetic_X)
            y_resampled_list.append(synthetic_y)

        X_out = pd.DataFrame(np.vstack(X_resampled_list), columns=X.columns)
        y_out = pd.Series(np.concatenate(y_resampled_list), name=y.name)
        return X_out, y_out


# ==============================================================================
# TAHAP 1: EKSPLORASI DAN PEMISAHAN DOMAIN FITUR (EDA)
# ==============================================================================
def load_and_prepare_eda(
    filepath: str = "students_dropout_academic_success.csv",
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    print("=" * 75)
    print("TAHAP 1: EKSPLORASI DATA & PEMISAHAN DOMAIN FITUR (EDA)")
    print("=" * 75)

    # 1. Load Data (Deteksi otomatis delimiter ';' atau ',')
    with open(filepath, "r", encoding="utf-8") as f:
        sep = ";" if ";" in f.readline() else ","
    df = pd.read_csv(filepath, sep=sep)

    # Normalisasi nama kolom minor agar 100% sesuai standar dokumentasi UCI
    df = df.rename(
        columns={
            "Marital Status": "Marital status",
            "Daytime/evening attendance\t": "Daytime/evening attendance",
            "target": "Target",
        }
    )

    print(f"Total Data Awal (Sebelum Filter): {df.shape[0]} baris x {df.shape[1]} kolom")
    for label, count in df["Target"].value_counts().items():
        print(f"  - {label:<10}: {count} ({count / len(df) * 100:.2f}%)")

    # 2. Hapus status 'Enrolled' untuk fokus pada Binary Classification tegas
    df = df[df["Target"] != "Enrolled"].copy().reset_index(drop=True)

    # 3. Label Encoding untuk Target: Dropout = 1, Graduate = 0
    df["Target_Binary"] = df["Target"].map({"Dropout": 1, "Graduate": 0})
    df = df.drop(columns=["Target"])

    # 4. Definisi List Pemisahan Domain Fitur (36 Fitur Independen)
    feature_domains = {
        "demography": [
            "Marital status",
            "Nacionality",
            "Displaced",
            "Gender",
            "Age at enrollment",
            "International",
        ],
        "socioeconomic": [
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
        ],
        "admission": [
            "Application mode",
            "Application order",
            "Course",
            "Daytime/evening attendance",
            "Previous qualification",
            "Previous qualification (grade)",
            "Admission grade",
        ],
        "academic_s1": [col for col in df.columns if "1st sem" in col],
        "academic_s2": [col for col in df.columns if "2nd sem" in col],
    }

    # 5. Sanity Check Eksekusi
    print(f"\nTotal Data setelah drop 'Enrolled': {df.shape}")
    dist = df["Target_Binary"].value_counts(normalize=True) * 100
    counts = df["Target_Binary"].value_counts()
    print(f"  - Graduate (0): {counts[0]} mahasiswa ({dist[0]:.2f}%)")
    print(f"  - Dropout  (1): {counts[1]} mahasiswa ({dist[1]:.2f}%)")

    missing_total = int(df.isnull().sum().sum())
    assert missing_total == 0, "Ditemukan missing values, butuh imputasi!"
    print(f"Cek Missing Values: {missing_total} (Aman, 0 missing values)")

    print("\nRingkasan Pemisahan Domain Fitur:")
    for domain, cols in feature_domains.items():
        print(f"  - {domain:<15}: {len(cols)} fitur")

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

    # 1. Identifikasi Tipe Fitur: Kategorikal Nominal vs Biner vs Numerik
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

    numeric_cols = [
        c for c in X.columns if c not in categorical_nominal_cols + binary_cols
    ]

    print(f"Klasifikasi Tipe Kolom ({X.shape[1]} Fitur Independen):")
    print(f"  - Kategorikal Nominal (One-Hot Encoding) : {len(categorical_nominal_cols)} fitur")
    print(f"  - Biner (0/1 Passthrough)                : {len(binary_cols)} fitur")
    print(f"  - Numerik (StandardScaler)               : {len(numeric_cols)} fitur")

    # 2. TRAIN-TEST SPLIT TERLEBIH DAHULU (SEBELUM ENCODING & SMOTE!)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train_raw = X_train.reset_index(drop=True)
    X_test_raw = X_test.reset_index(drop=True)
    y_train_raw = y_train.reset_index(drop=True)
    y_test_raw = y_test.reset_index(drop=True)

    print("\n[1] Hasil Stratified Train-Test Split (Sebelum Encoding & Balancing):")
    print(
        f"  - X_train_raw : {X_train_raw.shape} | "
        f"Graduate(0): {(y_train_raw == 0).sum()}, Dropout(1): {(y_train_raw == 1).sum()}"
    )
    print(
        f"  - X_test_raw  : {X_test_raw.shape}  | "
        f"Graduate(0): {(y_test_raw == 0).sum()}, Dropout(1): {(y_test_raw == 1).sum()}"
    )

    # 3. ONE-HOT ENCODING & SCALING (FIT HANYA PADA X_TRAIN)
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

    X_train_encoded_arr = preprocessor.fit_transform(X_train_raw)
    X_test_encoded_arr = preprocessor.transform(X_test_raw)

    feature_names_out = list(preprocessor.get_feature_names_out())
    X_train_encoded = pd.DataFrame(X_train_encoded_arr, columns=feature_names_out)
    X_test_encoded = pd.DataFrame(X_test_encoded_arr, columns=feature_names_out)

    print("\n[2] Hasil One-Hot Encoding & Scaling (Fit hanya pada X_train):")
    print(f"  - Dimensi X_train_encoded : {X_train_encoded.shape}")
    print(f"  - Dimensi X_test_encoded  : {X_test_encoded.shape}")

    # 4. BALANCING DENGAN SMOTE (HANYA PADA DATA TRAINING!)
    smote = SklearnSMOTE(k_neighbors=5, random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_encoded, y_train_raw)

    print("\n[3] Hasil Balancing dengan SMOTE (HANYA pada Train Set):")
    print(
        f"  - Sebelum SMOTE (y_train)       : "
        f"Graduate(0) = {(y_train_raw == 0).sum()}, Dropout(1) = {(y_train_raw == 1).sum()} (Total: {len(y_train_raw)})"
    )
    print(
        f"  - Sesudah SMOTE (y_train_smote) : "
        f"Graduate(0) = {(y_train_smote == 0).sum()}, Dropout(1) = {(y_train_smote == 1).sum()} (Total: {len(y_train_smote)})"
    )
    print(
        f"  - Test Set (y_test - MURNI)     : "
        f"Graduate(0) = {(y_test_raw == 0).sum()}, Dropout(1) = {(y_test_raw == 1).sum()} (Total: {len(y_test_raw)})"
    )

    # 5. SIMPAN KE SUBFOLDER TERSTRUKTUR DI `processed_data/`
    dir_raw_split = os.path.join(output_dir, "1_raw_split")
    dir_encoded = os.path.join(output_dir, "2_encoded")
    dir_smote = os.path.join(output_dir, "3_train_smote")

    for d in [dir_raw_split, dir_encoded, dir_smote]:
        os.makedirs(d, exist_ok=True)

    # Subfolder 1: 1_raw_split (Sebelum Encoding - 36 Fitur Asli)
    X_train_raw.to_csv(os.path.join(dir_raw_split, "X_train_raw.csv"), index=False)
    y_train_raw.to_frame(name=target_col).to_csv(os.path.join(dir_raw_split, "y_train.csv"), index=False)
    X_test_raw.to_csv(os.path.join(dir_raw_split, "X_test_raw.csv"), index=False)
    y_test_raw.to_frame(name=target_col).to_csv(os.path.join(dir_raw_split, "y_test.csv"), index=False)

    # Subfolder 2: 2_encoded (Sesudah Encoding & Scaling - 227 Fitur, Sebelum SMOTE)
    X_train_encoded.to_csv(os.path.join(dir_encoded, "X_train_encoded.csv"), index=False)
    y_train_raw.to_frame(name=target_col).to_csv(os.path.join(dir_encoded, "y_train.csv"), index=False)
    X_test_encoded.to_csv(os.path.join(dir_encoded, "X_test_encoded.csv"), index=False)
    y_test_raw.to_frame(name=target_col).to_csv(os.path.join(dir_encoded, "y_test.csv"), index=False)
    joblib.dump(preprocessor, os.path.join(dir_encoded, "preprocessor.joblib"))

    # Subfolder 3: 3_train_smote (Data Latih Seimbang 50:50 Siap Training Model)
    X_train_smote.to_csv(os.path.join(dir_smote, "X_train_smote.csv"), index=False)
    y_train_smote.to_frame(name=target_col).to_csv(os.path.join(dir_smote, "y_train_smote.csv"), index=False)

    print(f"\n[4] Seluruh data tersimpan rapi per tahapan di '{output_dir}/':")
    print("  ├── 1_raw_split/   (X_train_raw.csv, y_train.csv, X_test_raw.csv, y_test.csv)")
    print("  ├── 2_encoded/     (X_train_encoded.csv, X_test_encoded.csv, y_test.csv, preprocessor.joblib)")
    print("  └── 3_train_smote/ (X_train_smote.csv, y_train_smote.csv)")
    print("=" * 75)


if __name__ == "__main__":
    df_clean, _ = load_and_prepare_eda("students_dropout_academic_success.csv")
    preprocess_and_split_no_leakage(df_clean)
