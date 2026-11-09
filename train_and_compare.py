"""
================================================================================
PEMODELAN & PERBANDINGAN ALGORITMA — DECISION TREE VS RANDOM FOREST
================================================================================
File ini disusun dengan gaya modular yang sama seperti `preprocess_data.py` agar
mudah dipelajari dan dipresentasikan sesuai spesifikasi tugas:

  - Tujuan      : Memprediksi nilai / IPK akhir Semester 2 dari data awal semester
  - Dua Model   : Decision Tree Regressor vs. Random Forest Regressor
  - Evaluasi    : RMSE, MAE, dan R² (Skala Nilai 0-20 & Skala IPK 0.00-4.00)
  - Penjelasan  : Feature Importance dengan SHAP (Horizontal Bar Plot)

Tahapan Fungsi:
  1. load_processed_data()     -> Memuat X_train, X_test, y_train, y_test dari processed_data/
  2. build_preprocessor()      -> Menyusun ColumnTransformer (OneHotEncoder + SimpleImputer)
  3. compare_dt_vs_rf()        -> Melatih & membandingkan Decision Tree vs Random Forest
  4. export_shap_and_plots()   -> Membuat grafik perbandingan model & SHAP Bar Plot
  5. save_model_artifacts()    -> Menyimpan model terbaik & menguji prediksi 1 mahasiswa
================================================================================
"""

import json
import os
from typing import Any, Dict, List, Tuple

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from predictor import FEATURE_LABELS_ID, load_artifacts, map_transformed_to_original_feature, predict_and_explain
from preprocess_data import BINARY_COLS, NOMINAL_CATEGORICAL_COLS, NUMERIC_COLS, RANDOM_STATE, TARGET_COL

plt.rcParams["figure.figsize"] = (10, 6)
sns.set_theme(style="whitegrid", palette="deep")


# ==============================================================================
# BAGIAN A: MEMUAT DATA BERSIH DARI PROCESSED_DATA/
# ==============================================================================
def load_processed_data(data_dir: str = "processed_data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    print("=" * 75)
    print("[BAGIAN A] MEMUAT DATASET BERSIH DARI PROCESSED_DATA/")
    print("=" * 75)

    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))[TARGET_COL].astype(float)
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))[TARGET_COL].astype(float)

    print(f"Data Latih (Train) : {X_train.shape[0]} mahasiswa x {X_train.shape[1]} fitur")
    print(f"Data Uji   (Test)  : {X_test.shape[0]} mahasiswa x {X_test.shape[1]} fitur")
    print(f"Target Prediksi    : '{TARGET_COL}' (Skala 0-20 -> Konversi IPK 0.00-4.00)")

    return X_train, X_test, y_train, y_test


# ==============================================================================
# BAGIAN B: MEMBANGUN PREPROCESSOR DI DALAM PIPELINE
# ==============================================================================
def build_preprocessor() -> ColumnTransformer:
    """
    Menyusun ColumnTransformer untuk fitur numerik, kategori nominal, dan biner.
    Catatan: Decision Tree & Random Forest tidak membutuhkan StandardScaler.
    """
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False),
            ),
        ]
    )
    binary_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLS),
            ("cat", categorical_pipe, NOMINAL_CATEGORICAL_COLS),
            ("bin", binary_pipe, BINARY_COLS),
        ],
        verbose_feature_names_out=False,
    )


# ==============================================================================
# BAGIAN C: PELATIHAN & PERBANDINGAN DUA MODEL (DECISION TREE VS RANDOM FOREST)
# ==============================================================================
def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Menghitung metrik RMSE, MAE, dan R² pada skala asli (0-20) dan skala IPK (0-4)."""
    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)

    # Metrik Skala Asli (0 - 20)
    train_rmse = root_mean_squared_error(y_train, pred_train)
    test_rmse = root_mean_squared_error(y_test, pred_test)
    train_mae = mean_absolute_error(y_train, pred_train)
    test_mae = mean_absolute_error(y_test, pred_test)
    train_r2 = r2_score(y_train, pred_train)
    test_r2 = r2_score(y_test, pred_test)

    # Metrik Skala IPK (0.00 - 4.00), di mana IPK = Nilai / 5.0
    test_rmse_ipk = test_rmse / 5.0
    test_mae_ipk = test_mae / 5.0

    return {
        "Model": model_name,
        "Train_RMSE": round(float(train_rmse), 4),
        "Test_RMSE": round(float(test_rmse), 4),
        "Train_MAE": round(float(train_mae), 4),
        "Test_MAE": round(float(test_mae), 4),
        "Train_R2": round(float(train_r2), 4),
        "Test_R2": round(float(test_r2), 4),
        "Test_RMSE_IPK": round(float(test_rmse_ipk), 4),
        "Test_MAE_IPK": round(float(test_mae_ipk), 4),
    }


def compare_dt_vs_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Pipeline, Pipeline, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    print("\n" + "=" * 75)
    print("[BAGIAN C] PERBANDINGAN DUA MODEL: DECISION TREE VS RANDOM FOREST")
    print("=" * 75)

    cv_5fold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 1. Decision Tree Regressor (5-Fold CV Tuning agar tidak overfitting)
    print("1. Melatih & mengoptimasi Decision Tree Regressor (5-Fold CV)...")
    dt_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", DecisionTreeRegressor(random_state=RANDOM_STATE)),
        ]
    )
    dt_param_grid = {
        "model__max_depth": [4, 6, 8, 10],
        "model__min_samples_leaf": [5, 10, 20, 30],
        "model__min_samples_split": [10, 20, 40],
        "model__ccp_alpha": [0.0, 0.005, 0.01, 0.02],
    }
    dt_search = GridSearchCV(
        estimator=dt_pipeline,
        param_grid=dt_param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv_5fold,
        n_jobs=-1,
        refit=True,
    )
    dt_search.fit(X_train, y_train)
    best_dt = dt_search.best_estimator_
    print(f"   Hyperparameter Terbaik Decision Tree: {dt_search.best_params_}")

    # 2. Random Forest Regressor (5-Fold CV Tuning)
    print("2. Melatih & mengoptimasi Random Forest Regressor (5-Fold CV)...")
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    rf_param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [6, 8, 10, 14],
        "model__min_samples_leaf": [3, 5, 10],
        "model__max_features": ["sqrt", 0.5, 1.0],
    }
    rf_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv_5fold,
        n_jobs=-1,
        refit=True,
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print(f"   Hyperparameter Terbaik Random Forest: {rf_search.best_params_}")

    # Evaluasi Kedua Model
    res_dt = evaluate_model("Decision Tree Regressor", best_dt, X_train, y_train, X_test, y_test)
    res_rf = evaluate_model("Random Forest Regressor", best_rf, X_train, y_train, X_test, y_test)

    comparison_df = pd.DataFrame([res_dt, res_rf])

    print("\n--- HASIL PERBANDINGAN METRIK EVALUASI (DATA UJI / TEST SET) ---")
    display_cols = ["Model", "Test_RMSE", "Test_MAE", "Test_R2", "Test_RMSE_IPK", "Test_MAE_IPK"]
    print(comparison_df[display_cols].to_string(index=False))

    winner = "Random Forest Regressor" if res_rf["Test_R2"] >= res_dt["Test_R2"] else "Decision Tree Regressor"
    print(f"\n>> KESIMPULAN: '{winner}' unggul dengan R² = {max(res_rf['Test_R2'], res_dt['Test_R2']):.4f} "
          f"dan rata-rata selisih error (MAE) hanya ±{min(res_rf['Test_MAE_IPK'], res_dt['Test_MAE_IPK']):.2f} poin IPK.")

    return best_dt, best_rf, comparison_df, dt_search.best_params_, rf_search.best_params_


# ==============================================================================
# BAGIAN D: VISUALISASI PERBANDINGAN MODEL & SHAP FEATURE IMPORTANCE (BAR PLOT)
# ==============================================================================
def export_shap_and_plots(
    best_rf: Pipeline,
    comparison_df: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: str = "model",
) -> pd.DataFrame:
    print("\n" + "=" * 75)
    print("[BAGIAN D] VISUALISASI PERBANDINGAN & SHAP FEATURE IMPORTANCE (BAR PLOT)")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Grafik Perbandingan Decision Tree vs Random Forest
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = comparison_df["Model"]
    x_pos = np.arange(len(models))
    width = 0.35

    axes[0].bar(x_pos - width / 2, comparison_df["Test_RMSE"], width, label="RMSE (Skala 0-20)", color="#2B6CB0")
    axes[0].bar(x_pos + width / 2, comparison_df["Test_MAE"], width, label="MAE (Skala 0-20)", color="#4FD1C5")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(models, fontweight="bold")
    axes[0].set_ylabel("Nilai Error (Semakin Kecil Semakin Baik)")
    axes[0].set_title("Perbandingan Error: RMSE & MAE", fontweight="bold")
    axes[0].legend()

    bars_r2 = axes[1].bar(models, comparison_df["Test_R2"], color=["#ED8936", "#38A169"], width=0.45)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Skor R² (Mendekati 1.0 Semakin Baik)")
    axes[1].set_title("Perbandingan Akurasi Koefisien Determinasi (R²)", fontweight="bold")
    for bar in bars_r2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.4f}", ha="center", fontweight="bold")

    plt.tight_layout()
    comp_plot_path = os.path.join(output_dir, "perbandingan_dt_vs_rf.png")
    fig.savefig(comp_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"1. Grafik perbandingan model disimpan ke : {comp_plot_path}")

    # 2. Perhitungan SHAP TreeExplainer & Agregasi One-Hot ke 31 Fitur Asli
    preprocessor = best_rf.named_steps["preprocessor"]
    regressor = best_rf.named_steps["model"]
    explainer = shap.TreeExplainer(regressor)

    X_test_trans = preprocessor.transform(X_test)
    transformed_cols = list(preprocessor.get_feature_names_out())
    shap_values_matrix = explainer.shap_values(X_test_trans)

    original_features = list(X_test.columns)
    agg_shap_df = pd.DataFrame(0.0, index=np.arange(len(X_test)), columns=original_features)

    for idx_col, t_col in enumerate(transformed_cols):
        orig_col = map_transformed_to_original_feature(t_col, original_features)
        agg_shap_df[orig_col] += shap_values_matrix[:, idx_col]

    # Rata-rata nilai absolut SHAP per fitur (Global Feature Importance)
    mean_abs_shap = agg_shap_df.abs().mean(axis=0).sort_values(ascending=False)
    shap_summary_df = pd.DataFrame({
        "feature": mean_abs_shap.index,
        "feature_label": [FEATURE_LABELS_ID.get(f, f) for f in mean_abs_shap.index],
        "mean_abs_shap_grade": mean_abs_shap.values.round(4),
        "mean_abs_shap_ipk": (mean_abs_shap.values / 5.0).round(4),
    })

    print("\n--- TOP 10 FAKTOR PALING BERPENGARUH (SHAP GLOBAL IMPORTANCE) ---")
    print(shap_summary_df.head(10).to_string(index=False))

    # Plot Horizontal Bar Plot Top 12 Fitur SHAP
    top_n = 12
    top_shap = shap_summary_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_shap["feature_label"], top_shap["mean_abs_shap_grade"], color="#2B6CB0", edgecolor="none")
    ax.set_xlabel("Rata-rata Kontribusi Absolut SHAP |SHAP Value| (Poin Nilai Skala 0-20)", fontweight="bold")
    ax.set_title("Feature Importance dengan SHAP (Top 12 Faktor Penentu Performa Akademik)", fontsize=12, fontweight="bold")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.03, bar.get_y() + bar.get_height() / 2, f"+/- {w:.2f} poin", va="center", fontsize=9)
    ax.set_xlim(0, top_shap["mean_abs_shap_grade"].max() * 1.18)

    plt.tight_layout()
    shap_plot_path = os.path.join(output_dir, "shap_feature_importance_barplot.png")
    fig.savefig(shap_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n2. Grafik SHAP Bar Plot disimpan ke      : {shap_plot_path}")

    return shap_summary_df


# ==============================================================================
# BAGIAN E: SIMPAN ARTEFAK MODEL & UJI PREDIKSI 1 MAHASISWA
# ==============================================================================
def save_model_artifacts(
    best_rf: Pipeline,
    comparison_df: pd.DataFrame,
    dt_params: Dict[str, Any],
    rf_params: Dict[str, Any],
    shap_summary_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "model",
) -> None:
    print("\n" + "=" * 75)
    print("[BAGIAN E] PENYIMPANAN ARTEFAK MODEL & UJI PREDIKSI 1 MAHASISWA")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "final_pipeline.joblib")
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    demo_path = os.path.join(output_dir, "demo_students.json")
    comparison_csv_path = os.path.join(output_dir, "model_comparison.csv")

    joblib.dump(best_rf, model_path)
    comparison_df.to_csv(comparison_csv_path, index=False)

    # Skema fitur untuk validasi input aplikasi Streamlit
    feature_schema: Dict[str, Any] = {}
    default_feature_values: Dict[str, Any] = {}

    for col in X_train.columns:
        if col in NOMINAL_CATEGORICAL_COLS or col in BINARY_COLS:
            mode_val = int(X_train[col].mode().iloc[0])
            default_feature_values[col] = mode_val
            feature_schema[col] = {
                "type": "binary" if col in BINARY_COLS else "categorical",
                "categories": sorted([int(x) for x in X_train[col].dropna().unique()]),
                "default": mode_val,
            }
        else:
            med_val = float(X_train[col].median())
            default_feature_values[col] = round(med_val, 3)
            feature_schema[col] = {
                "type": "numeric",
                "min": round(float(X_train[col].min()), 3),
                "max": round(float(X_train[col].max()), 3),
                "median": round(med_val, 3),
            }

    rf_row = comparison_df[comparison_df["Model"] == "Random Forest Regressor"].iloc[0].to_dict()
    dt_row = comparison_df[comparison_df["Model"] == "Decision Tree Regressor"].iloc[0].to_dict()

    metadata_payload = {
        "project": "Prediksi IPK Akhir Semester & Penjelasan Faktor SHAP (Proyek P08)",
        "task_type": "regression",
        "target_column": TARGET_COL,
        "target_scale": [0.0, 20.0],
        "gpa_scale": [0.0, 4.0],
        "random_state": RANDOM_STATE,
        "compared_models": ["Decision Tree Regressor", "Random Forest Regressor"],
        "selected_model": "Random Forest Regressor (Tuned 5-Fold CV)",
        "best_hyperparameters": rf_params,
        "decision_tree_hyperparameters": dt_params,
        "metrics": {
            "train_rmse": rf_row["Train_RMSE"],
            "test_rmse": rf_row["Test_RMSE"],
            "train_mae": rf_row["Train_MAE"],
            "test_mae": rf_row["Test_MAE"],
            "train_r2": rf_row["Train_R2"],
            "test_r2": rf_row["Test_R2"],
            "test_rmse_ipk": rf_row["Test_RMSE_IPK"],
            "test_mae_ipk": rf_row["Test_MAE_IPK"],
        },
        "comparison_table": [dt_row, rf_row],
        "top_shap_features": shap_summary_df.head(10).to_dict(orient="records"),
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "shap": shap.__version__,
        },
        "feature_names_in": list(X_train.columns),
        "numeric_cols": NUMERIC_COLS,
        "nominal_categorical_cols": NOMINAL_CATEGORICAL_COLS,
        "binary_cols": BINARY_COLS,
        "default_feature_values": default_feature_values,
        "feature_schema": feature_schema,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2, ensure_ascii=False)

    # Buat 4 sampel mahasiswa untuk demo
    test_with_pred = X_test.copy()
    test_with_pred["_pred"] = best_rf.predict(X_test)
    test_with_pred["_actual"] = y_test.values

    high_case = test_with_pred.sort_values("_pred", ascending=False).iloc[0]
    mid_case = test_with_pred.iloc[(test_with_pred["_pred"] - 12.0).abs().argsort().iloc[0]]
    low_case = test_with_pred[test_with_pred["curricular_units_1st_sem_grade"] > 0].sort_values("_pred", ascending=True).iloc[0]
    extreme_case = test_with_pred.sort_values("_pred", ascending=True).iloc[0]

    demo_payload = {}
    for label_key, row_series in [
        ("performa_tinggi", high_case),
        ("performa_sedang", mid_case),
        ("risiko_tinggi", low_case),
        ("kasus_ekstrem_nol", extreme_case),
    ]:
        clean_dict = {}
        for col in X_train.columns:
            val = row_series[col]
            clean_dict[col] = int(val) if col in (NOMINAL_CATEGORICAL_COLS + BINARY_COLS) else round(float(val), 3)
        demo_payload[label_key] = clean_dict

    with open(demo_path, "w", encoding="utf-8") as f:
        json.dump(demo_payload, f, indent=2, ensure_ascii=False)

    print("Artefak berhasil disimpan di folder 'model/':")
    print(f"  1. {model_path}")
    print(f"  2. {metadata_path}")
    print(f"  3. {demo_path}")
    print(f"  4. {comparison_csv_path}")

    # Verifikasi fungsi predict_and_explain() untuk 1 mahasiswa
    load_artifacts.cache_clear()
    hasil = predict_and_explain(demo_payload["performa_sedang"], top_k=5)
    print("\n--- CONTOH INFERENSI 1 MAHASISWA (INPUT -> PREDIKSI IPK + PENJELASAN SHAP) ---")
    print(f"Prediksi Nilai Akhir (Skala 0-20) : {hasil['predicted_grade']:.2f} / 20.00")
    print(f"Prediksi IPK Akhir   (Skala 0-4)  : {hasil['predicted_gpa_4']:.2f} / 4.00")
    print(f"Penjelasan Ringkas Dosen PA       : {hasil['summary_text']}")
    print("=" * 75)


def main():
    X_train, X_test, y_train, y_test = load_processed_data("processed_data")
    _, best_rf, comparison_df, dt_params, rf_params = compare_dt_vs_rf(X_train, y_train, X_test, y_test)
    shap_summary_df = export_shap_and_plots(best_rf, comparison_df, X_test, output_dir="model")
    save_model_artifacts(
        best_rf=best_rf,
        comparison_df=comparison_df,
        dt_params=dt_params,
        rf_params=rf_params,
        shap_summary_df=shap_summary_df,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        output_dir="model",
    )


if __name__ == "__main__":
    main()
