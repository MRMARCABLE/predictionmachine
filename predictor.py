"""
================================================================================
SISTEM PRESKRIPTIF DETEKSI DINI RISIKO DROP-OUT MAHASISWA
MENGGUNAKAN RANDOM FOREST DAN SHAP EXPLAINABILITY
--------------------------------------------------------------------------------
TAHAP 4 & TAHAP 5: INTEGRASI MESIN PENJELAS (SHAP TREEEXPLAINER)
DAN PEMBANGUNAN RULE-ENGINE REKOMENDASI INTERVENSI (LOGIKA BISNIS)
================================================================================
Modul ini memuat model Random Forest dan Preprocessor yang telah diserialisasi,
kemudian menyediakan:
  1. `explain_single_student()` -> Menghitung probabilitas Dropout & mengekstrak
     Top 3 fitur dengan nilai SHAP absolut tertinggi menggunakan shap.TreeExplainer.
  2. `generate_recommendations()` -> Rule-Engine (if-else & dictionary mapping)
     yang menerjemahkan Top 3 fitur SHAP + probabilitas DO menjadi kalimat aksi
     preskriptif bagi pihak kampus / Dosen Pembimbing Akademik (PA).
================================================================================
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import joblib
import numpy as np
import pandas as pd
import shap

# ==============================================================================
# DAFTAR 36 FITUR ASLI & LABEL BAHASA INDONESIA UNTUK INTERPRETASI
# ==============================================================================
CATEGORICAL_NOMINAL_COLS = [
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

FEATURE_LABELS_ID: Dict[str, str] = {
    # A. Demografi
    "Marital status": "Status Pernikahan",
    "Nacionality": "Kewarganegaraan",
    "Displaced": "Mahasiswa Rantau (Displaced)",
    "Gender": "Jenis Kelamin",
    "Age at enrollment": "Usia Saat Mendaftar",
    "International": "Mahasiswa Internasional",
    # B. Sosial-Ekonomi & Finansial
    "Mother's qualification": "Pendidikan Ibu",
    "Father's qualification": "Pendidikan Ayah",
    "Mother's occupation": "Pekerjaan Ibu",
    "Father's occupation": "Pekerjaan Ayah",
    "Educational special needs": "Kebutuhan Khusus Pendidikan",
    "Debtor": "Status Memiliki Tunggakan/Hutang (Debtor)",
    "Tuition fees up to date": "Kepatuhan Pembayaran UKT/SPP Tepat Waktu",
    "Scholarship holder": "Status Penerima Beasiswa",
    "Unemployment rate": "Tingkat Pengangguran Makro",
    "Inflation rate": "Tingkat Inflasi Makro",
    "GDP": "Pertumbuhan PDB (GDP)",
    # C. Akademik Awal / Admisi
    "Application mode": "Jalur Pendaftaran Masuk",
    "Application order": "Urutan Pilihan Program Studi",
    "Course": "Program Studi (Jurusan)",
    "Daytime/evening attendance": "Waktu Perkuliahan (Pagi/Malam)",
    "Previous qualification": "Kualifikasi Pendidikan Sebelumnya",
    "Previous qualification (grade)": "Nilai Pendidikan Sebelumnya",
    "Admission grade": "Nilai Ujian Masuk (Admission Grade)",
    # D1. Akademik Semester 1
    "Curricular units 1st sem (credited)": "SKS Diakui/Konversi Semester 1",
    "Curricular units 1st sem (enrolled)": "SKS Diambil Semester 1",
    "Curricular units 1st sem (evaluations)": "Jumlah Evaluasi Ujian Semester 1",
    "Curricular units 1st sem (approved)": "Jumlah Mata Kuliah/SKS Lulus Semester 1",
    "Curricular units 1st sem (grade)": "Rata-rata Nilai Semester 1",
    "Curricular units 1st sem (without evaluations)": "Mata Kuliah Tanpa Evaluasi Semester 1",
    # D2. Akademik Semester 2
    "Curricular units 2nd sem (credited)": "SKS Diakui/Konversi Semester 2",
    "Curricular units 2nd sem (enrolled)": "SKS Diambil Semester 2",
    "Curricular units 2nd sem (evaluations)": "Jumlah Evaluasi Ujian Semester 2",
    "Curricular units 2nd sem (approved)": "Jumlah Mata Kuliah/SKS Lulus Semester 2",
    "Curricular units 2nd sem (grade)": "Rata-rata Nilai Semester 2",
    "Curricular units 2nd sem (without evaluations)": "Mata Kuliah Tanpa Evaluasi Semester 2",
}


# ==============================================================================
# 1. PEMUATAN MODEL, PREPROCESSOR & INISIALISASI SHAP TREEEXPLAINER
# ==============================================================================
class DropoutPrescriptivePredictor:
    """
    Kelas terpadu untuk melakukan prediksi risiko Drop-Out, ekstraksi penjelasan
    SHAP (Top 3 Fitur), dan pembuatan rekomendasi preskriptif berbasis aturan.
    """

    def __init__(
        self,
        model_path: str = "model/random_forest_dropout.joblib",
        preprocessor_path: str = "processed_data/2_encoded/preprocessor.joblib",
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File model tidak ditemukan di: {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"File preprocessor tidak ditemukan di: {preprocessor_path}")

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.encoded_feature_names: List[str] = list(self.preprocessor.get_feature_names_out())

        # Inisialisasi shap.TreeExplainer khusus untuk arsitektur pohon (Random Forest)
        self.explainer = shap.TreeExplainer(self.model)

    @staticmethod
    def _map_encoded_col_to_original(encoded_col: str) -> str:
        """
        Memetakan nama kolom hasil One-Hot Encoding (misal 'Course_9500' atau
        'Marital status_1') kembali ke nama fitur induk aslinya ('Course', 'Marital status').
        """
        for cat_col in CATEGORICAL_NOMINAL_COLS:
            prefix = f"{cat_col}_"
            if encoded_col.startswith(prefix):
                return cat_col
        return encoded_col

    def _extract_dropout_shap_vector(self, X_encoded_row: pd.DataFrame) -> np.ndarray:
        """
        Mengekstrak vektor nilai SHAP (panjang 227) khusus untuk kelas positif
        Dropout (label = 1) pada satu baris data mahasiswa.
        """
        shap_vals = self.explainer.shap_values(X_encoded_row)

        # Penanganan kompatibilitas berbagai versi library SHAP:
        # - Jika list [shap_class_0, shap_class_1], ambil indeks 1 (Dropout)
        # - Jika array 3D (n_samples, n_features, n_classes), ambil [0, :, 1]
        if isinstance(shap_vals, list):
            return np.asarray(shap_vals[1][0], dtype=float)
        shap_arr = np.asarray(shap_vals, dtype=float)
        if shap_arr.ndim == 3:
            return shap_arr[0, :, 1]
        return shap_arr[0]

    # ==========================================================================
    # TAHAP 4: FUNGSI PREDIKSI & PENJELASAN TOP 3 FITUR SHAP (1 MAHASISWA)
    # ==========================================================================
    def explain_single_student(
        self,
        student_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Menerima input 1 baris data mahasiswa (36 fitur asli), lalu mengembalikan:
          - Probabilitas prediksi Dropout & Graduate
          - Label hasil prediksi & tingkat risiko (LOW / MEDIUM / HIGH)
          - Daftar Top-K (default 3) fitur dengan nilai SHAP absolut tertinggi
          - Rekomendasi intervensi dari Rule-Engine (Tahap 5)
        """
        if isinstance(student_data, dict):
            df_row = pd.DataFrame([student_data])
        elif isinstance(student_data, pd.Series):
            df_row = student_data.to_frame().T
        elif isinstance(student_data, pd.DataFrame):
            df_row = student_data.iloc[[0]].copy()
        else:
            raise TypeError("Input student_data harus berupa dict, pd.Series, atau pd.DataFrame.")

        # 1. Transformasi 36 fitur mentah -> 227 fitur terenkode menggunakan preprocessor
        X_encoded_arr = self.preprocessor.transform(df_row)
        X_encoded_df = pd.DataFrame(X_encoded_arr, columns=self.encoded_feature_names)

        # 2. Hitung probabilitas prediksi dari Random Forest
        proba = self.model.predict_proba(X_encoded_df)[0]
        prob_graduate = float(proba[0])
        prob_dropout = float(proba[1])
        pred_code = int(prob_dropout >= 0.50)
        pred_label = "Dropout" if pred_code == 1 else "Graduate"

        if prob_dropout >= 0.70:
            risk_level = "HIGH (Risiko Sangat Tinggi)"
        elif prob_dropout >= 0.40:
            risk_level = "MEDIUM (Risiko Sedang / Waspada)"
        else:
            risk_level = "LOW (Risiko Rendah / Aman)"

        # 3. Hitung nilai SHAP untuk kelas Dropout (1) dan agregasikan kembali ke 36 fitur asli
        shap_vec_227 = self._extract_dropout_shap_vector(X_encoded_df)

        aggregated_shap: Dict[str, float] = {}
        for enc_col, s_val in zip(self.encoded_feature_names, shap_vec_227):
            orig_col = self._map_encoded_col_to_original(enc_col)
            aggregated_shap[orig_col] = aggregated_shap.get(orig_col, 0.0) + float(s_val)

        # 4. Urutkan fitur berdasarkan nilai absolut SHAP tertinggi (|SHAP|)
        sorted_features = sorted(
            aggregated_shap.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        raw_dict = df_row.iloc[0].to_dict()
        top_shap_features: List[Dict[str, Any]] = []

        for rank, (feat_name, shap_val) in enumerate(sorted_features[:top_k], start=1):
            raw_val = raw_dict.get(feat_name, None)
            if isinstance(raw_val, (np.integer, np.floating)):
                raw_val = raw_val.item()

            direction = (
                "Mendorong Risiko Dropout (+)"
                if shap_val > 0
                else "Menahan Risiko / Mendorong Lulus (-)"
            )
            top_shap_features.append(
                {
                    "rank": rank,
                    "feature_name": feat_name,
                    "feature_label_id": FEATURE_LABELS_ID.get(feat_name, feat_name),
                    "raw_value": raw_val,
                    "shap_value": round(float(shap_val), 4),
                    "abs_shap_value": round(abs(float(shap_val)), 4),
                    "impact_direction": direction,
                }
            )

        # 5. Panggil Rule-Engine Rekomendasi (Tahap 5)
        recommendations = generate_recommendations(
            prob_dropout=prob_dropout,
            top_shap_features=top_shap_features,
            student_raw_data=raw_dict,
        )

        return {
            "status": "success",
            "prediction": {
                "label": pred_label,
                "label_code": pred_code,
                "probability_dropout": round(prob_dropout, 4),
                "probability_dropout_pct": round(prob_dropout * 100, 2),
                "probability_graduate": round(prob_graduate, 4),
                "probability_graduate_pct": round(prob_graduate * 100, 2),
                "risk_level": risk_level,
            },
            "top_3_shap_features": top_shap_features,
            "recommendations": recommendations,
        }


# ==============================================================================
# TAHAP 5: PEMBANGUNAN RULE-ENGINE REKOMENDASI (LOGIKA BISNIS)
# ==============================================================================
def generate_recommendations(
    prob_dropout: float,
    top_shap_features: List[Dict[str, Any]],
    student_raw_data: Dict[str, Any],
) -> List[str]:
    """
    Menerjemahkan 3 fitur teratas dari SHAP beserta nilai aktual mahasiswa dan
    probabilitas Dropout menjadi kalimat aksi intervensi preskriptif.
    """
    recommendations: List[str] = []

    # Kamus pemetaan aturan rekomendasi berbasis fitur & kondisi aktual mahasiswa
    for feat_info in top_shap_features:
        feat = feat_info["feature_name"]
        val = student_raw_data.get(feat)
        shap_val = feat_info["shap_value"]

        # ----------------------------------------------------------------------
        # 1. ATURAN DOMAIN FINANSIAL & SOSIAL-EKONOMI
        # ----------------------------------------------------------------------
        if feat == "Debtor":
            if int(val) == 1 and prob_dropout > 0.70:
                recommendations.append(
                    "Rekomendasi: Jadwalkan pertemuan dengan bagian keuangan untuk restrukturisasi pembayaran."
                )
            elif int(val) == 1:
                recommendations.append(
                    "Rekomendasi: Lakukan konseling finansial dini terkait status hutang/tunggakan biaya pendidikan mahasiswa."
                )

        elif feat == "Tuition fees up to date":
            if int(val) == 0 and prob_dropout > 0.70:
                recommendations.append(
                    "Rekomendasi: Hubungi mahasiswa segera melalui bagian keuangan untuk skema cicilan UKT/SPP atau penangguhan pembayaran."
                )
            elif int(val) == 0:
                recommendations.append(
                    "Rekomendasi: Kirimkan pengingat pembayaran UKT/SPP dan tawarkan konsultasi bantuan biaya kuliah."
                )

        elif feat == "Scholarship holder":
            if int(val) == 0 and shap_val > 0:
                recommendations.append(
                    "Rekomendasi: Arahkan mahasiswa ke bagian Kemahasiswaan untuk pendaftaran beasiswa bantuan internal atau keringanan UKT."
                )

        # ----------------------------------------------------------------------
        # 2. ATURAN DOMAIN PERFORMA AKADEMIK SEMESTER 1 & SEMESTER 2
        # ----------------------------------------------------------------------
        elif feat in [
            "Curricular units 2nd sem (approved)",
            "Curricular units 1st sem (approved)",
        ]:
            sem_label = "Semester 2" if "2nd sem" in feat else "Semester 1"
            if float(val) <= 3 or shap_val > 0:
                recommendations.append(
                    f"Rekomendasi: Wajibkan sesi konseling dengan Dosen Pembimbing Akademik (PA) karena rendahnya kelulusan mata kuliah pada {sem_label} (lulus: {int(val)} MK) serta evaluasi beban SKS."
                )

        elif feat in [
            "Curricular units 2nd sem (grade)",
            "Curricular units 1st sem (grade)",
        ]:
            sem_label = "Semester 2" if "2nd sem" in feat else "Semester 1"
            if float(val) < 11.0 or shap_val > 0:
                recommendations.append(
                    f"Rekomendasi: Masukkan mahasiswa ke dalam program Academic Mentoring / Tutorial Sebaya untuk memperbaiki nilai rata-rata {sem_label} (nilai saat ini: {float(val):.2f})."
                )

        elif feat in [
            "Curricular units 2nd sem (evaluations)",
            "Curricular units 1st sem (evaluations)",
            "Curricular units 2nd sem (without evaluations)",
            "Curricular units 1st sem (without evaluations)",
        ]:
            if shap_val > 0:
                recommendations.append(
                    "Rekomendasi: Lakukan pemantauan presensi perkuliahan dan pastikan mahasiswa mengikuti ujian akhir semester secara penuh."
                )

        elif feat in [
            "Curricular units 2nd sem (enrolled)",
            "Curricular units 1st sem (enrolled)",
        ]:
            if shap_val > 0:
                recommendations.append(
                    "Rekomendasi: Tinjau ulang pengambilan KRS bersama Dosen Wali agar jumlah SKS yang diambil sesuai dengan kapasitas akademik mahasiswa."
                )

        # ----------------------------------------------------------------------
        # 3. ATURAN DOMAIN ADMISI & DEMOGRAFI
        # ----------------------------------------------------------------------
        elif feat in ["Admission grade", "Previous qualification (grade)"]:
            if shap_val > 0:
                recommendations.append(
                    "Rekomendasi: Berikan kelas matrikulasi atau penguatan materi dasar program studi untuk menjembatani kesiapan akademik awal."
                )

        elif feat == "Age at enrollment":
            if float(val) >= 23 and shap_val > 0:
                recommendations.append(
                    "Rekomendasi: Tawarkan fleksibilitas jadwal bimbingan akademik dan dukungan manajemen waktu bagi mahasiswa berusia dewasa/bekerja."
                )

        elif feat == "Course":
            if shap_val > 0:
                recommendations.append(
                    "Rekomendasi: Lakukan asesmen minat-bakat dan pendampingan adaptasi kurikulum pada Program Studi yang sedang ditempuh."
                )

    # Jika mahasiswa berisiko rendah (Graduate) atau belum ada aturan spesifik yang terpicu
    if not recommendations:
        if prob_dropout >= 0.50:
            recommendations.append(
                "Rekomendasi: Jadwalkan evaluasi berkala bersama Dosen Pembimbing Akademik (PA) untuk memantau faktor risiko utama dari analisis SHAP."
            )
        else:
            recommendations.append(
                "Rekomendasi: Pertahankan performa akademik dan status administrasi saat ini; mahasiswa berada pada jalur kelulusan yang aman."
            )

    # Hilangkan duplikasi kalimat rekomendasi sambil menjaga urutan prioritas SHAP
    unique_recommendations: List[str] = []
    for rec in recommendations:
        if rec not in unique_recommendations:
            unique_recommendations.append(rec)

    return unique_recommendations


# ==============================================================================
# DEMONSTRASI & PENGUJIAN OTOMATIS PADA SAMPEL MAHASISWA ASLI
# ==============================================================================
def main() -> None:
    print("=" * 80)
    print("TAHAP 4 & 5: PENGUJIAN MESIN PENJELAS SHAP & RULE-ENGINE REKOMENDASI")
    print("=" * 80)

    predictor = DropoutPrescriptivePredictor()

    # Load data uji mentah (36 fitur asli sebelum encoding)
    X_test_raw = pd.read_csv("processed_data/1_raw_split/X_test_raw.csv")
    y_test = pd.read_csv("processed_data/1_raw_split/y_test.csv")["Target_Binary"]

    # Kasus Uji 1: Mahasiswa dari Test Set (Indeks 117) dengan masalah finansial (Debtor == 1 & UKT menunggak)
    student_case_1 = X_test_raw.loc[117].to_dict()

    # Kasus Uji 2: Mahasiswa berprestasi dari Test Set yang Aktual Graduate
    grad_mask = (
        (X_test_raw["Tuition fees up to date"] == 1)
        & (X_test_raw["Debtor"] == 0)
        & (X_test_raw["Curricular units 2nd sem (approved)"] >= 6)
        & (y_test == 0)
    )
    sample_grad_idx = X_test_raw[grad_mask].index[0]
    student_case_2 = X_test_raw.loc[sample_grad_idx].to_dict()

    for idx_case, (title, student_dict) in enumerate(
        [
            ("KASUS 1: Mahasiswa Berisiko Tinggi (Memiliki Tunggakan/Debtor=1)", student_case_1),
            ("KASUS 2: Mahasiswa Berisiko Rendah (Akademik & Finansial Lancar)", student_case_2),
        ],
        start=1,
    ):
        result = predictor.explain_single_student(student_dict)
        pred = result["prediction"]

        print(f"\n[{idx_case}] {title}")
        print("-" * 80)
        print(f"  * Prediksi Status       : {pred['label']} (Kode: {pred['label_code']})")
        print(f"  * Probabilitas Dropout  : {pred['probability_dropout_pct']:.2f}% ({pred['probability_dropout']:.4f})")
        print(f"  * Probabilitas Graduate : {pred['probability_graduate_pct']:.2f}% ({pred['probability_graduate']:.4f})")
        print(f"  * Kategori Risiko       : {pred['risk_level']}")
        print("\n  * Top 3 Fitur dengan Nilai SHAP Absolut Tertinggi:")
        for f in result["top_3_shap_features"]:
            print(
                f"    {f['rank']}. {f['feature_name']} ({f['feature_label_id']})\n"
                f"       - Nilai Aktual Mahasiswa : {f['raw_value']}\n"
                f"       - Nilai SHAP             : {f['shap_value']:+.4f} (|SHAP| = {f['abs_shap_value']:.4f})\n"
                f"       - Arah Kontribusi        : {f['impact_direction']}"
            )
        print("\n  * Output Rule-Engine Rekomendasi:")
        for rec in result["recommendations"]:
            print(f"    -> {rec}")

    demo_output = predictor.explain_single_student(student_case_1)
    sample_json_path = "model/sample_prediction_output.json"
    with open(sample_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_student_36_features": student_case_1,
                "output_response": demo_output,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print(f"Contoh payload request & response JSON disimpan di: {sample_json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
