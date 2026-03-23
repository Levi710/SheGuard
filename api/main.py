"""
MamaGuard -- FastAPI Server
Maternal risk prediction API with clinical safety net and explainability.
"""

import torch
import numpy as np
import pickle
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.schemas import PredictionRequest, PredictionResponse, AlertTier
from api.alert_logic import compute_alert_tier, generate_action_text
from src.model import MamaGuardMamba3
from src.explainability import explain_prediction
from api.extract_report import router as extract_router

# --- App setup ----------------------------------------------------------------
app = FastAPI(
    title="SheGuard -- Maternal Risk API",
    description=(
        "Predicts maternal mortality risk from prenatal visit sequences. "
        "Built with Mamba3 SSM for deployment in low-resource clinics."
    ),
    version="1.0.0",
)

app.include_router(extract_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model loading ────────────────────────────────────────────────────────────

MODEL_PATH  = "models/mamaguard_mamba3.pt"
SCALER_PATH = "models/scaler.pkl"

model  = None
scaler = None
device = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_ORDER = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
FEATURE_DEFAULTS = {
    'Age': 30.0, 'SystolicBP': 120.0, 'DiastolicBP': 80.0,
    'BS': 7.5, 'BodyTemp': 36.8, 'HeartRate': 76.0
}


@app.on_event("startup")
async def load_model():
    """Load model and scaler at server startup."""
    global model, scaler

    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model not found at {MODEL_PATH}. Run training first.")
        return

    model = MamaGuardMamba3(input_dim=6, d_model=64, n_layers=4, n_classes=3, d_state=32)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    print(f"MamaGuard model loaded on {device}")


# --- Routes -------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    return {
        "status":       "healthy",
        "model_loaded": model is not None,
        "device":       device,
        "version":      "1.0.0",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Process patient visit data and return risk prediction with explanation.
    """

    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Contact system administrator."
        )

    # Prepare visit data
    visits = request.visits
    visit_map = {
        'Age': 'age', 'SystolicBP': 'systolic_bp', 'DiastolicBP': 'diastolic_bp',
        'BS': 'blood_sugar', 'BodyTemp': 'body_temp', 'HeartRate': 'heart_rate'
    }

    raw_array = []
    missing_counts = []

    for visit in visits:
        row = []
        missing = 0
        for feat in FEATURE_ORDER:
            attr = visit_map[feat]
            val  = getattr(visit, attr, None)
            if val is None:
                val = FEATURE_DEFAULTS[feat]
                missing += 1
            row.append(val)
        raw_array.append(row)
        missing_counts.append(missing)

    raw_np = np.array(raw_array, dtype=np.float32)

    # Data quality score
    total_fields  = len(visits) * len(FEATURE_ORDER)
    missing_total = sum(missing_counts)
    data_quality  = round(1.0 - missing_total / total_fields, 3)

    # Scale
    scaled_np = scaler.transform(raw_np)

    # Pad or truncate to SEQ_LEN=5
    SEQ_LEN = 5
    n = len(scaled_np)
    if n < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - n, scaled_np.shape[1]), dtype=np.float32)
        scaled_np = np.vstack([pad, scaled_np])
    elif n > SEQ_LEN:
        scaled_np = scaled_np[-SEQ_LEN:]

    # Clinical safety net (hard WHO rules)
    from api.alert_logic import apply_clinical_safety_net, AlertTier

    visits_raw = [v.model_dump() for v in request.visits]
    forced_tier, forced_reason = apply_clinical_safety_net(visits_raw)

    # Run model and explain
    explanation = explain_prediction(model, scaled_np, scaler, device)

    if forced_tier is not None:
        explanation["top_reasons"].insert(0, f"[WHO guideline] {forced_reason}")

    # Alert tier
    alert_tier, suppressed = compute_alert_tier(
        probabilities    = explanation["probabilities"],
        patient_id       = request.patient_id,
        visit_importances= explanation["visit_importance"],
    )

    # Override with clinical rule minimum if needed
    tier_order = {AlertTier.GREEN: 0, AlertTier.AMBER: 1, AlertTier.RED: 2}
    if forced_tier is not None:
        if tier_order[forced_tier] > tier_order[alert_tier]:
            alert_tier = forced_tier
            suppressed = False

    # Action text
    action, transfer_order = generate_action_text(
        tier             = alert_tier,
        top_reasons      = explanation["top_reasons"],
        data_quality     = data_quality,
        staff_available  = request.staff_available,
        blood_units      = request.blood_units,
    )

    return PredictionResponse(
        patient_id        = request.patient_id,
        risk_level        = explanation["risk_level"],
        alert_tier        = alert_tier,
        confidence        = explanation["confidence"],
        data_quality      = data_quality,
        top_reasons       = explanation["top_reasons"],
        feature_importance= explanation["feature_importance"],
        visit_importance  = explanation["visit_importance"],
        action_required   = action,
        transfer_order    = transfer_order,
        suppressed        = suppressed,
        probabilities     = explanation["probabilities"],
    )


@app.get("/stats")
async def get_stats():
    """Returns patient assessment statistics."""
    from api.alert_logic import _alert_history
    total   = len(_alert_history)
    by_tier = {"GREEN": 0, "AMBER": 0, "RED": 0}
    for v in _alert_history.values():
        by_tier[v["tier"].value] += 1

    return {
        "patients_assessed": total,
        "by_tier": by_tier,
        "model_version": "SheGuard-Mamba3-v1",
    }


# --- Dashboard static files ---------------------------------------------------

DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")

if os.path.isdir(DASHBOARD_DIR):
    app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="dashboard-static")


@app.get("/")
async def serve_dashboard():
    """Serve the dashboard HTML at root."""
    index_path = os.path.join(DASHBOARD_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"status": "ok", "model_loaded": model is not None, "device": device}