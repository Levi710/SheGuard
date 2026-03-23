"""
MamaGuard — Explainability
Gradient-based attribution for per-feature and per-visit importance.
"""

import torch
import torch.nn.functional as F
import numpy as np
from src.model import MamaGuardMamba3

FEATURE_NAMES = ['Age', 'SystolicBP', 'DiastolicBP', 'BloodSugar', 'BodyTemp', 'HeartRate']
RISK_LABELS   = ['Low risk', 'Medium risk', 'High risk']
RISK_EMOJI    = ['🟢', '🟡', '🔴']


def explain_prediction(
    model: MamaGuardMamba3,
    x_sequence: np.ndarray,
    scaler,
    device: str = "cpu"
) -> dict:
    """
    Runs the model on one patient and returns an explanation dict with:
    risk_level, probabilities, confidence, top_reasons,
    feature_importance, and visit_importance.
    """
    model.eval()

    x_tensor = torch.tensor(x_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    x_tensor.requires_grad_(True)

    # Forward pass
    logits = model(x_tensor)
    probs  = F.softmax(logits, dim=-1)

    pred_class = probs.argmax(dim=-1).item()
    confidence = probs[0, pred_class].item()

    # Gradient attribution
    score = logits[0, pred_class]
    score.backward()

    grads = x_tensor.grad[0].cpu().numpy()
    attribution = np.abs(grads)

    # Per-feature importance (average over visits)
    feature_importance = attribution.mean(axis=0)
    feature_importance = feature_importance / (feature_importance.sum() + 1e-9)

    # Per-visit importance (average over features)
    visit_importance = attribution.mean(axis=1)
    visit_importance = visit_importance / (visit_importance.sum() + 1e-9)

    # Build human-readable top reasons
    sorted_features = sorted(
        zip(FEATURE_NAMES, feature_importance),
        key=lambda x: x[1], reverse=True
    )
    top_reasons = []

    x_orig = scaler.inverse_transform(x_sequence)

    for feat, importance in sorted_features[:2]:
        feat_idx = FEATURE_NAMES.index(feat)
        vals = x_orig[:, feat_idx]
        trend = vals[-1] - vals[0]

        if abs(trend) > 0.5:
            direction = "rising" if trend > 0 else "falling"
            top_reasons.append(
                f"{feat} is {direction} (from {vals[0]:.1f} to {vals[-1]:.1f})"
            )
        else:
            top_reasons.append(
                f"{feat} is consistently elevated (avg {vals.mean():.1f})"
            )

    # Assemble result
    probs_np = probs[0].detach().cpu().numpy()

    return {
        "risk_level":         RISK_LABELS[pred_class],
        "risk_emoji":         RISK_EMOJI[pred_class],
        "probabilities": {
            label: round(float(p), 4)
            for label, p in zip(RISK_LABELS, probs_np)
        },
        "confidence":        round(confidence, 4),
        "top_reasons":       top_reasons,
        "feature_importance": {
            feat: round(float(imp), 4)
            for feat, imp in zip(FEATURE_NAMES, feature_importance)
        },
        "visit_importance":  [round(float(v), 4) for v in visit_importance],
    }