"""
================================================================================
MODUL KONTRAK SERAH TERIMA ML ENGINEER -> APP DEVELOPER (PROYEK P08)
================================================================================
Menyediakan fungsi `predict_and_explain(data_satu_mahasiswa)` yang memuat
Pipeline final (`model/final_pipeline.joblib`), menghitung prediksi nilai
akademik (Regresi), dan mengekstrak faktor penjelas lokal menggunakan SHAP
(`TreeExplainer`) yang sudah diagregasi kembali ke nama fitur asli.
================================================================================
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd
import shap

DEFAULT_MODEL_PATH = os.path.join("model", "final_pipeline.joblib")
DEFAULT_METADATA_PATH = os.path.join("model", "model_metadata.json")

# Label bahasa Indonesia yang ramah pengguna untuk aplikasi dosen PA
FEATURE_LABELS_ID: Dict[str, str] = {
    "marital_status": "Status Pernikahan",
    "application_mode": "Jalur Pendaftaran",
    "application_order": "Urutan Pilihan Jurusan",
    "course": "Program Studi (Kode Jurusan)",
    "daytime_evening_attendance": "Kelas Pagi / Malam",
    "previous_qualification": "Kualifikasi Pendidikan Sebelumnya",
    "previous_qualification_grade": "Nilai Pendidikan Sebelumnya",
    "nationality": "Kewarganegaraan",
    "mothers_qualification": "Pendidikan Ibu",
    "fathers_qualification": "Pendidikan Ayah",
    "mothers_occupation": "Pekerjaan Ibu",
    "fathers_occupation": "Pekerjaan Ayah",
    "admission_grade": "Nilai Ujian Masuk (Admission Grade)",
    "displaced": "Mahasiswa Perantau (Displaced)",
    "educational_special_needs": "Kebutuhan Khusus Pendidikan",
    "debtor": "Memiliki Tunggakan (Debtor)",
    "tuition_fees_up_to_date": "UKT / SPP Lancar",
    "gender": "Jenis Kelamin",
    "scholarship_holder": "Penerima Beasiswa",
    "age_at_enrollment": "Usia Saat Mendaftar",
    "international": "Mahasiswa Internasional",
    "curricular_units_1st_sem_credited": "SKS Diakui (Sem 1)",
    "curricular_units_1st_sem_enrolled": "SKS Diambil (Sem 1)",
    "curricular_units_1st_sem_evaluations": "Jumlah Evaluasi Ujian (Sem 1)",
    "curricular_units_1st_sem_approved": "SKS Lulus (Sem 1)",
    "curricular_units_1st_sem_grade": "Nilai Rata-Rata Semester 1",
    "curricular_units_1st_sem_without_evaluations": "SKS Tanpa Evaluasi (Sem 1)",
    "unemployment_rate": "Tingkat Pengangguran Makro (%)",
    "inflation_rate": "Tingkat Inflasi Makro (%)",
    "gdp": "Pertumbuhan GDP Makro (%)",
    "approval_rate_1st_sem": "Rasio Kelulusan SKS Semester 1",
}


@lru_cache(maxsize=2)
def load_artifacts(
    model_path: str = DEFAULT_MODEL_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
) -> Dict[str, Any]:
    """Memuat Pipeline model, metadata fitur, dan menginisialisasi SHAP TreeExplainer."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"File model tidak ditemukan di '{model_path}'. "
            "Jalankan seluruh sel di notebook `ml_engineer_regression.ipynb` terlebih dahulu."
        )

    pipeline = joblib.load(model_path)
    metadata: Dict[str, Any] = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(regressor)

    return {
        "pipeline": pipeline,
        "preprocessor": preprocessor,
        "regressor": regressor,
        "explainer": explainer,
        "metadata": metadata,
    }


def map_transformed_to_original_feature(
    transformed_col: str, original_features: List[str]
) -> str:
    """
    Memetakan nama kolom hasil ColumnTransformer/OneHotEncoder kembali ke nama
    fitur aslinya (contoh: 'course_9254' -> 'course').
    """
    if transformed_col in original_features:
        return transformed_col

    # Urutkan dari nama terpanjang agar prefix yang mirip tidak salah cocok
    for orig in sorted(original_features, key=len, reverse=True):
        if transformed_col.startswith(orig + "_"):
            return orig
    return transformed_col


def _prepare_single_row_df(
    data_satu_mahasiswa: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    expected_features: List[str],
    default_values: Dict[str, Any],
) -> pd.DataFrame:
    """Mengubah input dict/Series/DataFrame menjadi DataFrame 1 baris sesuai urutan kolom."""
    if isinstance(data_satu_mahasiswa, pd.DataFrame):
        if len(data_satu_mahasiswa) == 0:
            raise ValueError("DataFrame input kosong.")
        row_dict = data_satu_mahasiswa.iloc[0].to_dict()
    elif isinstance(data_satu_mahasiswa, pd.Series):
        row_dict = data_satu_mahasiswa.to_dict()
    elif isinstance(data_satu_mahasiswa, dict):
        row_dict = dict(data_satu_mahasiswa)
    else:
        raise TypeError(
            "Input `data_satu_mahasiswa` harus berupa dict, pandas.Series, atau pandas.DataFrame."
        )

    # Isi otomatis fitur `approval_rate_1st_sem` jika belum dihitung oleh form aplikasi
    if (
        "approval_rate_1st_sem" in expected_features
        and "approval_rate_1st_sem" not in row_dict
    ):
        enrolled = float(row_dict.get("curricular_units_1st_sem_enrolled", 0) or 0)
        approved = float(row_dict.get("curricular_units_1st_sem_approved", 0) or 0)
        row_dict["approval_rate_1st_sem"] = approved / enrolled if enrolled > 0 else 0.0

    # Pastikan semua kolom yang dibutuhkan model tersedia
    clean_row: Dict[str, Any] = {}
    for col in expected_features:
        if col in row_dict and row_dict[col] is not None and not pd.isna(row_dict[col]):
            clean_row[col] = row_dict[col]
        else:
            clean_row[col] = default_values.get(col, 0)

    return pd.DataFrame([clean_row], columns=expected_features)


def predict_and_explain(
    data_satu_mahasiswa: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    model_path: str = DEFAULT_MODEL_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Memprediksi nilai akhir akademik 1 mahasiswa dan mengembalikan faktor penjelas SHAP.

    Parameters
    ----------
    data_satu_mahasiswa : dict | pd.Series | pd.DataFrame
        Data fitur 1 mahasiswa (nama kolom asli sebelum One-Hot Encoding).
    model_path : str
        Lokasi file `.joblib` Pipeline final.
    metadata_path : str
        Lokasi file `.json` metadata model.
    top_k : int
        Jumlah faktor penjelas terpenting yang dikembalikan pada `top_factors`.

    Returns
    -------
    dict
        Berisi:
        - `predicted_grade`: float (prediksi nilai pada skala asli 0.0 - 20.0)
        - `predicted_gpa_4`: float (ekuivalen skala IPK 0.00 - 4.00)
        - `base_value`: float (nilai rata-rata dasar dari model)
        - `top_factors`: list of dict (faktor utama yang menaikkan/menurunkan prediksi)
        - `all_factors`: list of dict (seluruh fitur diurutkan berdasarkan |SHAP|)
        - `summary_text`: str (1 kalimat penjelasan dalam bahasa awam untuk dosen PA)
    """
    artifacts = load_artifacts(model_path=model_path, metadata_path=metadata_path)
    pipeline = artifacts["pipeline"]
    preprocessor = artifacts["preprocessor"]
    explainer = artifacts["explainer"]
    metadata = artifacts["metadata"]

    expected_features: List[str] = metadata.get(
        "feature_names_in", list(preprocessor.feature_names_in_)
    )
    default_values: Dict[str, Any] = metadata.get("default_feature_values", {})

    X_one = _prepare_single_row_df(
        data_satu_mahasiswa=data_satu_mahasiswa,
        expected_features=expected_features,
        default_values=default_values,
    )

    # 1. Prediksi nilai akhir
    raw_pred = float(pipeline.predict(X_one)[0])
    predicted_grade = float(np.clip(raw_pred, 0.0, 20.0))
    predicted_gpa_4 = round((predicted_grade / 20.0) * 4.0, 2)

    # 2. Transformasi fitur & hitung nilai SHAP lokal
    X_trans = preprocessor.transform(X_one)
    transformed_cols = list(preprocessor.get_feature_names_out())

    shap_values_raw = explainer.shap_values(X_trans)
    if isinstance(shap_values_raw, list):
        shap_row = np.asarray(shap_values_raw[0]).flatten()
    else:
        shap_row = np.asarray(shap_values_raw).flatten()

    expected_val = explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)):
        base_value = float(np.asarray(expected_val).flatten()[0])
    else:
        base_value = float(expected_val)

    # 3. Agregasi nilai SHAP dari pecahan One-Hot Encoding kembali ke fitur asli
    aggregated_shap: Dict[str, float] = {feat: 0.0 for feat in expected_features}
    for col_name, val in zip(transformed_cols, shap_row):
        orig_feat = map_transformed_to_original_feature(col_name, expected_features)
        aggregated_shap[orig_feat] = aggregated_shap.get(orig_feat, 0.0) + float(val)

    # 4. Susun daftar faktor diurutkan dari pengaruh absolut terbesar
    sorted_features = sorted(
        aggregated_shap.items(), key=lambda item: abs(item[1]), reverse=True
    )

    all_factors: List[Dict[str, Any]] = []
    for feat_name, shap_val in sorted_features:
        raw_val = X_one.iloc[0][feat_name]
        if isinstance(raw_val, (np.integer, int)):
            raw_val_clean: Any = int(raw_val)
        elif isinstance(raw_val, (np.floating, float)):
            raw_val_clean = round(float(raw_val), 3)
        else:
            raw_val_clean = raw_val

        direction = "menaikkan" if shap_val >= 0 else "menurunkan"
        all_factors.append(
            {
                "feature": feat_name,
                "feature_label": FEATURE_LABELS_ID.get(feat_name, feat_name),
                "raw_value": raw_val_clean,
                "shap_impact": round(float(shap_val), 4),
                "abs_impact": round(abs(float(shap_val)), 4),
                "direction": direction,
            }
        )

    top_factors = all_factors[:top_k]

    # 5. Buat 1 kalimat penjelasan bahasa awam sesuai Bab 5 Panduan P08
    pos_factors = [f for f in top_factors if f["shap_impact"] > 0.05]
    neg_factors = [f for f in top_factors if f["shap_impact"] < -0.05]

    parts: List[str] = [
        f"Prediksi nilai akhir mahasiswa adalah {predicted_grade:.2f}/20.00 "
        f"(setara IPK {predicted_gpa_4:.2f}/4.00, dari rata-rata dasar {base_value:.2f})"
    ]
    if pos_factors:
        top_pos = pos_factors[0]
        parts.append(
            f"terutama didorong naik oleh {top_pos['feature_label']} "
            f"({top_pos['raw_value']}, +{top_pos['shap_impact']:.2f} poin)"
        )
    if neg_factors:
        top_neg = neg_factors[0]
        parts.append(
            f"serta diturunkan oleh {top_neg['feature_label']} "
            f"({top_neg['raw_value']}, {top_neg['shap_impact']:.2f} poin)"
        )

    summary_text = ", ".join(parts) + "."

    return {
        "predicted_grade": round(predicted_grade, 2),
        "predicted_gpa_4": predicted_gpa_4,
        "base_value": round(base_value, 2),
        "top_factors": top_factors,
        "all_factors": all_factors,
        "summary_text": summary_text,
        "disclaimer": (
            "Catatan Etika: Hasil prediksi ini merupakan alat bantu pendukung keputusan "
            "bagi Dosen Pembimbing Akademik (PA), bukan vonis mutlak terhadap mahasiswa."
        ),
    }

