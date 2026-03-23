# src/evaluate.py
"""
MamaGuard — Model Evaluation
Produces accuracy, per-class metrics, confusion matrix, ROC-AUC, and a text report.

Usage: python -m src.evaluate
"""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from src.data_pipeline import load_and_preprocess
from src.model import MamaGuardMamba3

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH    = "data/maternal_health.csv"
MODEL_PATH  = "models/mamaguard_mamba3.pt"
SCALER_PATH = "models/scaler.pkl"
REPORT_PATH = "models/evaluation_report.txt"
CM_PATH     = "models/confusion_matrix.png"
ROC_PATH    = "models/roc_curves.png"

CLASS_NAMES  = ["Low risk", "Medium risk", "High risk"]
CLASS_COLORS = ["#2e7d32", "#e65100", "#c62828"]

# ── Load model ────────────────────────────────────────────────────────────────

def load_model(device: str):
    """Load trained model weights into a fresh MamaGuardMamba3 instance."""
    model = MamaGuardMamba3(
        input_dim=6, d_model=64, n_layers=4, n_classes=3, d_state=32
    )
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.to(device)
    model.eval()
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def get_predictions(model, X: np.ndarray, device: str, batch_size: int = 64):
    """
    Run the model on all validation sequences in batches.
    Returns: y_pred (N,), y_proba (N, 3)
    """
    model.eval()
    all_probs = []

    for i in range(0, len(X), batch_size):
        batch   = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    y_proba = np.vstack(all_probs)
    y_pred  = y_proba.argmax(axis=1)
    return y_pred, y_proba


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    """Plot raw and normalised confusion matrices side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("MamaGuard Confusion Matrix", fontsize=14, fontweight="bold")

    for ax_idx, (data, title) in enumerate([
        (cm, "Raw counts"),
        (cm.astype(float) / cm.sum(axis=1, keepdims=True), "Normalised (row %)")
    ]):
        im = axes[ax_idx].imshow(data, cmap="RdYlGn", vmin=0,
                                  vmax=(1 if ax_idx == 1 else None))
        axes[ax_idx].set_xticks(range(3))
        axes[ax_idx].set_yticks(range(3))
        axes[ax_idx].set_xticklabels(CLASS_NAMES, rotation=15, ha="right")
        axes[ax_idx].set_yticklabels(CLASS_NAMES)
        axes[ax_idx].set_xlabel("Predicted label")
        axes[ax_idx].set_ylabel("True label")
        axes[ax_idx].set_title(title)

        for i in range(3):
            for j in range(3):
                val  = data[i, j]
                text = f"{val:.2f}" if ax_idx == 1 else str(int(val))
                color = "white" if (ax_idx == 1 and val < 0.4) or \
                                   (ax_idx == 0 and val > cm.max() * 0.6) else "black"
                axes[ax_idx].text(j, i, text, ha="center", va="center",
                                   fontsize=11, color=color, fontweight="bold")

        plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved -> {save_path}")


# ── ROC curves ────────────────────────────────────────────────────────────────

def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, save_path: str):
    """Plot one-vs-rest ROC curves per class with AUC scores."""
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random (AUC = 0.50)")

    for i, (class_name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc         = roc_auc_score(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{class_name} (AUC = {auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — MamaGuard Mamba3", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curves saved -> {save_path}")


# ── Text report ───────────────────────────────────────────────────────────────

def build_text_report(
    y_true, y_pred, y_proba,
    class_report: str,
    cm: np.ndarray,
    n_train: int,
    n_val: int
) -> str:
    """Build a complete text report for model card / README."""
    overall_acc = accuracy_score(y_true, y_pred)

    from sklearn.preprocessing import label_binarize
    y_bin  = label_binarize(y_true, classes=[0, 1, 2])
    aucs   = [roc_auc_score(y_bin[:, i], y_proba[:, i]) for i in range(3)]

    hr_recall    = cm[2, 2] / max(cm[2, :].sum(), 1)
    hr_precision = cm[2, 2] / max(cm[:, 2].sum(), 1)

    report = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║              MAMAGUARD — MODEL EVALUATION REPORT                        ║
║              Generated automatically by src/evaluate.py                 ║
╚══════════════════════════════════════════════════════════════════════════╝

MODEL
  Architecture : MamaGuard-Mamba3 (Trapezoidal SSM + MIMO + Complex state)
  Parameters   : ~287,815
  Input        : Sequence of prenatal visits (up to 5) × 6 vital signs
  Output       : 3-class risk prediction (Low / Medium / High)
  Dataset      : UCI Maternal Health Risk (1,014 rows)
  Training set : {n_train} sequences
  Validation set: {n_val} sequences

──────────────────────────────────────────────────────────────────────────
PERFORMANCE METRICS (on held-out validation set)
──────────────────────────────────────────────────────────────────────────

  Overall accuracy     : {overall_acc:.3f} ({overall_acc*100:.1f}%)

  ROC-AUC per class:
    Low risk           : {aucs[0]:.3f}
    Medium risk        : {aucs[1]:.3f}
    High risk          : {aucs[2]:.3f}

  ** HIGH RISK RECALL (most critical for patient safety):
    {hr_recall:.3f} — of all truly high-risk patients, {hr_recall*100:.1f}% were correctly flagged

  ** HIGH RISK PRECISION (alarm fatigue indicator):
    {hr_precision:.3f} — of all patients flagged high-risk, {hr_precision*100:.1f}% truly were

  Detailed per-class report:
{class_report}

──────────────────────────────────────────────────────────────────────────
CONFUSION MATRIX (raw counts)
──────────────────────────────────────────────────────────────────────────
  Rows = true label, Columns = predicted label

                 Pred: Low   Pred: Mid   Pred: High
  True: Low        {cm[0,0]:5d}       {cm[0,1]:5d}        {cm[0,2]:5d}
  True: Mid        {cm[1,0]:5d}       {cm[1,1]:5d}        {cm[1,2]:5d}
  True: High       {cm[2,0]:5d}       {cm[2,1]:5d}        {cm[2,2]:5d}

  Most dangerous mistakes (False Negatives for High Risk):
    High-risk patients predicted as Low risk  : {cm[2,0]}
    High-risk patients predicted as Mid risk  : {cm[2,1]}

──────────────────────────────────────────────────────────────────────────
HYBRID SYSTEM NOTE
──────────────────────────────────────────────────────────────────────────
  MamaGuard uses a HYBRID architecture:
    1. Mamba3 model: handles subtle temporal patterns (learned from data)
    2. WHO clinical rules: hard overrides for obvious danger signs
       - Rule 1: SystolicBP >= 160 -> RED
       - Rule 2: SystolicBP >= 140 -> AMBER minimum
       - Rule 3: Blood sugar > 11.1 -> AMBER minimum
       - Rule 4: BP rise >= 20 mmHg -> AMBER minimum
       - Rule 5: 3+ vitals escalating simultaneously -> RED

  The metrics above reflect the NEURAL MODEL ONLY (without clinical rules).
  In deployment, the clinical rules provide an additional safety floor,
  meaning real-world recall for high-risk cases is higher than shown above.

──────────────────────────────────────────────────────────────────────────
LIMITATIONS
──────────────────────────────────────────────────────────────────────────
  1. SMALL DATASET: Trained on 1,014 rows from a single UCI dataset.
     Real-world clinical models typically require 10,000–100,000+ samples.

  2. SYNTHETIC SEQUENCES: The UCI dataset has no patient IDs or timestamps.
     We created artificial 5-visit sequences by sorting rows by age.
     These do not represent real patient trajectories.

  3. NOT CLINICALLY VALIDATED: This model has NOT been validated against
     real patient outcomes in a clinical setting. It must NOT be used
     for actual medical decisions without proper clinical validation trials.

  4. POPULATION BIAS: The UCI dataset was collected from a specific
     population. Performance may differ on patients from different
     regions, ethnicities, or healthcare contexts.

  5. RESEARCH PROTOTYPE: This is a proof-of-concept demonstrating the
     application of Mamba3 SSMs to maternal health risk prediction.
     The system design (alarm fatigue mitigation, resource-aware routing,
     OCR auto-fill) represents the primary contribution of this work.

──────────────────────────────────────────────────────────────────────────
CITATION
──────────────────────────────────────────────────────────────────────────
  If you use this work, please cite:
    MamaGuard: Maternal Mortality Early Warning using Mamba3 Sequential
    State-Space Models with Clinical Safety Rules.
    [Your Name], 2025. GitHub: [your-repo-url]
    Based on: Gu & Dao (2023) Mamba; UCI Maternal Health Risk dataset.

══════════════════════════════════════════════════════════════════════════
"""
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate():
    print("\n" + "="*60)
    print("  MamaGuard -- Model Evaluation")
    print("="*60 + "\n")

    for path, name in [(MODEL_PATH, "model"), (SCALER_PATH, "scaler"), (CSV_PATH, "dataset")]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at {path}")
            print("Run python -m src.train first.")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("\nLoading and preprocessing data...")
    (X_train, y_train, q_train,
     X_val, y_val, q_val,
     scaler) = load_and_preprocess(CSV_PATH)

    n_train = len(X_train)
    n_val   = len(X_val)

    print(f"Validation set: {n_val} sequences")
    print(f"Class distribution in validation: "
          f"Low={sum(y_val==0)} Mid={sum(y_val==1)} High={sum(y_val==2)}")

    # Load model and predict
    print("\nLoading model...")
    model = load_model(device)

    print("Running inference on validation set...")
    y_pred, y_proba = get_predictions(model, X_val, device)

    # Compute metrics
    print("\nComputing metrics...")
    overall_acc  = accuracy_score(y_val, y_pred)
    cm           = confusion_matrix(y_val, y_pred)
    class_report = classification_report(
        y_val, y_pred,
        target_names=CLASS_NAMES,
        digits=3
    )

    print(f"\n{'-'*50}")
    print(f"  Overall Accuracy: {overall_acc:.3f} ({overall_acc*100:.1f}%)")
    print(f"{'-'*50}")
    print("\nPer-class metrics:")
    print(class_report)

    print("Confusion matrix (rows=true, cols=predicted):")
    header = f"{'':15s} {'Low':>8s} {'Mid':>8s} {'High':>8s}"
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = f"  {name:13s} " + " ".join(f"{cm[i,j]:8d}" for j in range(3))
        print(row)

    hr_recall = cm[2, 2] / max(cm[2, :].sum(), 1)
    print(f"\n** HIGH RISK RECALL: {hr_recall:.3f}")
    if hr_recall < 0.60:
        print("   [!] WARNING: Less than 60% of high-risk patients detected by model alone.")
        print("   The WHO clinical rules provide the safety floor in deployment.")
    elif hr_recall < 0.75:
        print("   [!] Moderate. Consider retraining with more data (Path B).")
    else:
        print("   [OK] Good recall -- model is learning the high-risk pattern.")

    # Save plots
    print(f"\nSaving evaluation plots...")
    os.makedirs("models", exist_ok=True)
    plot_confusion_matrix(cm, CM_PATH)
    plot_roc_curves(y_val, y_proba, ROC_PATH)

    # Save text report
    report_text = build_text_report(
        y_val, y_pred, y_proba,
        class_report, cm, n_train, n_val
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  Full report saved -> {REPORT_PATH}")

    # Final summary
    print(f"\n{'='*60}")
    print("  EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Files saved:")
    print(f"    {REPORT_PATH}   <- paste into Hugging Face model card")
    print(f"    {CM_PATH}       <- include in LinkedIn post")
    print(f"    {ROC_PATH}      <- include in GitHub README")

    print(f"\n  Overall accuracy : {overall_acc*100:.1f}%")
    print(f"  High-risk recall : {hr_recall*100:.1f}%")

    if hr_recall < 0.60:
        print("\n  RECOMMENDATION: Retrain with augmented data before publishing.")
        print("  Current model relies heavily on WHO clinical rules for safety.")
        print("  This is still publishable as a research prototype -- be transparent.")
    else:
        print("\n  RECOMMENDATION: Model is ready to publish as research prototype.")
        print("  Include the evaluation_report.txt in your model card.")

    print()


if __name__ == "__main__":
    evaluate()