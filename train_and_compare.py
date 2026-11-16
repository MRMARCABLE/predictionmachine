"""
================================================================================
SISTEM PRESKRIPTIF DETEKSI DINI RISIKO DROP-OUT MAHASISWA
MENGGUNAKAN RANDOM FOREST DAN SHAP EXPLAINABILITY
--------------------------------------------------------------------------------
TAHAP 3: TRAINING MODEL, HYPERPARAMETER TUNING, EVALUASI & SERIALISASI
(DECISION TREE BASELINE VS RANDOM FOREST CLASSIFIER)
================================================================================
Prinsip Metodologi:
1. Data Training menggunakan data yang telah di-SMOTE (X_train_smote, y_train_smote)
   agar model tidak bias terhadap kelas mayoritas (Graduate).
2. Model Decision Tree Classifier dilatih sebagai baseline pembanding.
3. Model Random Forest Classifier dioptimasi parameter terbaiknya (n_estimators,
   max_depth, min_samples_split, min_samples_leaf) menggunakan GridSearchCV.
4. Evaluasi WAJIB dilakukan pada Test Set Murni (X_test_encoded, y_test) yang
   TIDAK tersentuh SMOTE menggunakan metrik Precision, Recall, F1-Score,
   Accuracy, dan ROC-AUC.
5. Model Random Forest optimal beserta metadata diserialisasi menggunakan joblib.
================================================================================
"""

import json
import os
from typing import Any, Dict, Tuple

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET_COL = "Target_Binary"


# ==============================================================================
# 1. LOAD DATA TRAINING (SMOTE) & DATA TESTING (MURNI / ASLI)
# ==============================================================================
def load_datasets(
    data_dir: str = "processed_data",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    print("=" * 75)
    print("TAHAP 3: TRAINING MODEL & SERIALISASI (DECISION TREE VS RANDOM FOREST)")
    print("=" * 75)

    smote_dir = os.path.join(data_dir, "3_train_smote")
    encoded_dir = os.path.join(data_dir, "2_encoded")

    X_train_smote = pd.read_csv(os.path.join(smote_dir, "X_train_smote.csv"))
    y_train_smote = pd.read_csv(os.path.join(smote_dir, "y_train_smote.csv"))[TARGET_COL].astype(int)

    X_test_encoded = pd.read_csv(os.path.join(encoded_dir, "X_test_encoded.csv"))
    y_test = pd.read_csv(os.path.join(encoded_dir, "y_test.csv"))[TARGET_COL].astype(int)

    print("[1] Ringkasan Dataset yang Dimuat:")
    print(
        f"  - Train Set (SMOTE) : {X_train_smote.shape} | "
        f"Graduate(0)={(y_train_smote == 0).sum()}, Dropout(1)={(y_train_smote == 1).sum()}"
    )
    print(
        f"  - Test Set  (MURNI) : {X_test_encoded.shape}  | "
        f"Graduate(0)={(y_test == 0).sum()}, Dropout(1)={(y_test == 1).sum()}"
    )

    return X_train_smote, y_train_smote, X_test_encoded, y_test


# ==============================================================================
# 2. LATIH BASELINE: DECISION TREE CLASSIFIER
# ==============================================================================
def train_decision_tree_baseline(
    X_train: pd.DataFrame, y_train: pd.Series
) -> DecisionTreeClassifier:
    print("\n[2] Melatih Baseline Model: Decision Tree Classifier...")
    dt_baseline = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt_baseline.fit(X_train, y_train)
    print(
        f"  - Decision Tree selesai dilatih (Kedalaman Pohon: {dt_baseline.get_depth()}, "
        f"Jumlah Daun: {dt_baseline.get_n_leaves()})"
    )
    return dt_baseline


# ==============================================================================
# 3. LATIH & TUNING: RANDOM FOREST CLASSIFIER (GRIDSEARCHCV)
# ==============================================================================
def train_and_tune_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[RandomForestClassifier, Dict[str, Any], float]:
    print("\n[3] Melatih & Mencari Parameter Terbaik: Random Forest Classifier (GridSearchCV)...")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 15, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        scoring="f1",  # Fokus optimasi pada keseimbangan Precision & Recall kelas Dropout (1)
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_f1 = float(grid_search.best_score_)

    print(f"  - Parameter Terbaik Ditemukan : {best_params}")
    print(f"  - Skor F1 Cross-Validation    : {best_cv_f1:.4f}")

    return best_rf, best_params, best_cv_f1


# ==============================================================================
# 4. EVALUASI MODEL PADA TEST SET MURNI (TIDAK DI-SMOTE)
# ==============================================================================
def evaluate_model_on_test(
    model: Any,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    # Prediksi pada Train (untuk cek indikasi overfitting) & Test Set Murni
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    train_f1 = float(f1_score(y_train, y_train_pred, pos_label=1))
    train_acc = float(accuracy_score(y_train, y_train_pred))

    precision_dropout = float(precision_score(y_test, y_test_pred, pos_label=1))
    recall_dropout = float(recall_score(y_test, y_test_pred, pos_label=1))
    f1_dropout = float(f1_score(y_test, y_test_pred, pos_label=1))
    f1_macro = float(f1_score(y_test, y_test_pred, average="macro"))
    accuracy = float(accuracy_score(y_test, y_test_pred))
    roc_auc = float(roc_auc_score(y_test, y_test_proba))
    cm = confusion_matrix(y_test, y_test_pred)

    print("\n" + "-" * 75)
    print(f"HASIL EVALUASI TEST SET MURNI: {model_name.upper()}")
    print("-" * 75)
    print(f"  * Precision (Kelas Dropout=1) : {precision_dropout:.4f} ({precision_dropout*100:.2f}%)")
    print(f"  * Recall    (Kelas Dropout=1) : {recall_dropout:.4f} ({recall_dropout*100:.2f}%)")
    print(f"  * F1-Score  (Kelas Dropout=1) : {f1_dropout:.4f} ({f1_dropout*100:.2f}%)")
    print(f"  * F1-Score  (Macro Average)   : {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    print(f"  * Accuracy  (Test Set Murni)  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  * ROC-AUC Score               : {roc_auc:.4f}")
    print(f"  * Perbandingan F1 Train vs Test: Train={train_f1:.4f} vs Test={f1_dropout:.4f}")
    print("\n  Confusion Matrix (Baris=Aktual, Kolom=Prediksi):")
    print(f"    - True Negative  (Graduate ditebak Graduate) : {cm[0, 0]}")
    print(f"    - False Positive (Graduate ditebak Dropout)  : {cm[0, 1]}")
    print(f"    - False Negative (Dropout ditebak Graduate)  : {cm[1, 0]}")
    print(f"    - True Positive  (Dropout ditebak Dropout)   : {cm[1, 1]}")
    print("\n  Classification Report Lengkap:")
    print(
        classification_report(
            y_test,
            y_test_pred,
            target_names=["Graduate (0)", "Dropout (1)"],
            digits=4,
        )
    )

    return {
        "model_name": model_name,
        "train_accuracy": train_acc,
        "train_f1_dropout": train_f1,
        "test_precision_dropout": precision_dropout,
        "test_recall_dropout": recall_dropout,
        "test_f1_dropout": f1_dropout,
        "test_f1_macro": f1_macro,
        "test_accuracy": accuracy,
        "test_roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
    }


# ==============================================================================
# 5. VISUALISASI KOMPARASI & SERIALISASI MODEL (JOBLIB)
# ==============================================================================
def save_plots_and_serialize(
    dt_model: DecisionTreeClassifier,
    rf_model: RandomForestClassifier,
    dt_metrics: Dict[str, Any],
    rf_metrics: Dict[str, Any],
    best_params: Dict[str, Any],
    best_cv_f1: float,
    feature_names: list,
    model_dir: str = "model",
) -> None:
    os.makedirs(model_dir, exist_ok=True)

    # 1. Serialisasi Model menggunakan Joblib
    rf_model_path = os.path.join(model_dir, "random_forest_dropout.joblib")
    dt_model_path = os.path.join(model_dir, "decision_tree_baseline.joblib")
    artifact_bundle_path = os.path.join(model_dir, "dropout_classifier_bundle.joblib")

    joblib.dump(rf_model, rf_model_path)
    joblib.dump(dt_model, dt_model_path)

    # Simpan juga bundel lengkap (model + preprocessor + feature_names + best_params)
    preprocessor_path = os.path.join("processed_data", "2_encoded", "preprocessor.joblib")
    preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None

    bundle = {
        "model": rf_model,
        "baseline_model": dt_model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "best_params": best_params,
        "metrics": {
            "random_forest": rf_metrics,
            "decision_tree_baseline": dt_metrics,
        },
        "target_mapping": {"Graduate": 0, "Dropout": 1},
    }
    joblib.dump(bundle, artifact_bundle_path)

    # 2. Simpan ringkasan evaluasi JSON
    summary_payload = {
        "best_model": "RandomForestClassifier",
        "best_hyperparameters": best_params,
        "cv_best_f1_score": best_cv_f1,
        "comparison_on_pure_test_set": {
            "DecisionTree_Baseline": dt_metrics,
            "RandomForest_Optimal": rf_metrics,
        },
    }
    metrics_json_path = os.path.join(model_dir, "evaluation_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    # 3. Buat Plot Komparasi Metrik & Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    metric_keys = ["test_precision_dropout", "test_recall_dropout", "test_f1_dropout", "test_accuracy", "test_roc_auc"]
    metric_labels = ["Precision\n(Dropout)", "Recall\n(Dropout)", "F1-Score\n(Dropout)", "Accuracy", "ROC-AUC"]
    dt_vals = [dt_metrics[k] for k in metric_keys]
    rf_vals = [rf_metrics[k] for k in metric_keys]

    x = np.arange(len(metric_labels))
    width = 0.36

    bars1 = axes[0].bar(x - width / 2, dt_vals, width, label="Decision Tree (Baseline)", color="#e07a5f")
    bars2 = axes[0].bar(x + width / 2, rf_vals, width, label="Random Forest (Optimal)", color="#2a9d8f")
    axes[0].set_ylim(0.60, 1.02)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metric_labels)
    axes[0].set_ylabel("Skor Evaluasi pada Test Set Murni")
    axes[0].set_title("Perbandingan Performa: Decision Tree vs Random Forest")
    axes[0].legend(loc="lower right")

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        axes[0].annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )

    cm_rf = np.array(rf_metrics["confusion_matrix"])
    sns.heatmap(
        cm_rf,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred: Graduate (0)", "Pred: Dropout (1)"],
        yticklabels=["Actual: Graduate (0)", "Actual: Dropout (1)"],
        ax=axes[1],
        cbar=False,
        annot_kws={"size": 13, "weight": "bold"},
    )
    axes[1].set_title("Confusion Matrix Random Forest (Test Set Murni)")

    plt.tight_layout()
    plot_path = os.path.join(model_dir, "model_comparison.png")
    plt.savefig(plot_path, dpi=180)
    plt.close(fig)

    print("\n" + "=" * 75)
    print("[4] SERIALISASI MODEL & ARTIFAK SELESAI")
    print("=" * 75)
    print(f"  1. Model Random Forest Optimal : {rf_model_path}")
    print(f"  2. Model Decision Tree Baseline: {dt_model_path}")
    print(f"  3. Bundel Lengkap Inference    : {artifact_bundle_path}")
    print(f"  4. Laporan Metrik JSON         : {metrics_json_path}")
    print(f"  5. Grafik Perbandingan & CM    : {plot_path}")
    print("=" * 75)


def main() -> None:
    X_train_smote, y_train_smote, X_test_encoded, y_test = load_datasets("processed_data")

    dt_baseline = train_decision_tree_baseline(X_train_smote, y_train_smote)
    best_rf, best_params, best_cv_f1 = train_and_tune_random_forest(X_train_smote, y_train_smote)

    dt_metrics = evaluate_model_on_test(
        dt_baseline,
        "Decision Tree (Baseline)",
        X_train_smote,
        y_train_smote,
        X_test_encoded,
        y_test,
    )
    rf_metrics = evaluate_model_on_test(
        best_rf,
        "Random Forest (Optimal Tuned)",
        X_train_smote,
        y_train_smote,
        X_test_encoded,
        y_test,
    )

    save_plots_and_serialize(
        dt_model=dt_baseline,
        rf_model=best_rf,
        dt_metrics=dt_metrics,
        rf_metrics=rf_metrics,
        best_params=best_params,
        best_cv_f1=best_cv_f1,
        feature_names=list(X_train_smote.columns),
        model_dir="model",
    )


if __name__ == "__main__":
    main()
