"""
MamaGuard — Alert Logic
Three-tier alerting with suppression, clinical safety net, and resource-aware routing.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict
from api.schemas import AlertTier

# ─── Thresholds ────────────────────────────────────────────────────────────────
RED_THRESHOLD   = 0.85
AMBER_THRESHOLD = 0.65
RED_CONSECUTIVE_VISITS = 2
SUPPRESSION_HOURS = 48
SUPPRESSION_DELTA = 0.15

# In-memory alert store (use Redis/DB in production)
_alert_history: Dict[str, dict] = {}


# ─── Clinical rule safety net ─────────────────────────────────────────────────

def apply_clinical_safety_net(visits_raw: list) -> tuple[str | None, str | None]:
    """
    Checks raw visit values against WHO clinical thresholds.
    Returns (forced_tier, reason) if a rule fires, else (None, None).
    """
    max_systolic  = max((v.get("systolic_bp")  or 0) for v in visits_raw)
    max_diastolic = max((v.get("diastolic_bp") or 0) for v in visits_raw)
    max_bs        = max((v.get("blood_sugar")  or 0) for v in visits_raw)

    latest     = visits_raw[-1]
    latest_sys = latest.get("systolic_bp") or 0
    latest_dia = latest.get("diastolic_bp") or 0

    # ── RED RULES ──────────────────────────────────────────────────────────────

    # Rule 1: Severe hypertension → RED
    if max_systolic >= 160 or max_diastolic >= 110:
        return "RED", (
            f"SystolicBP {max_systolic} mmHg meets WHO severe "
            f"hypertension threshold (≥160) — emergency referral required"
        )

    # Rule 5: Multi-vital simultaneous escalation → RED
    if len(visits_raw) >= 3:
        first = visits_raw[0]
        last  = visits_raw[-1]

        sys_rise  = (last.get("systolic_bp")  or 0) - (first.get("systolic_bp")  or 0)
        dia_rise  = (last.get("diastolic_bp") or 0) - (first.get("diastolic_bp") or 0)
        bs_rise   = (last.get("blood_sugar")  or 0) - (first.get("blood_sugar")  or 0)
        hr_rise   = (last.get("heart_rate")   or 0) - (first.get("heart_rate")   or 0)
        temp_rise = (last.get("body_temp")    or 0) - (first.get("body_temp")    or 0)

        escalating_count = 0
        if sys_rise  >= 15:  escalating_count += 1
        if dia_rise  >= 10:  escalating_count += 1
        if bs_rise   >= 3.0: escalating_count += 1
        if hr_rise   >= 15:  escalating_count += 1
        if temp_rise >= 0.5: escalating_count += 1

        if escalating_count >= 3:
            return "RED", (
                f"{escalating_count} vitals escalating simultaneously "
                f"(BP +{sys_rise:.0f} mmHg, HR +{hr_rise:.0f} bpm, "
                f"BS +{bs_rise:.1f} mmol/L) — combined deterioration "
                f"pattern, refer immediately"
            )

    # ── AMBER RULES ────────────────────────────────────────────────────────────

    # Rule 2: Hypertension in pregnancy → AMBER
    if latest_sys >= 140 or latest_dia >= 90:
        return "AMBER", (
            f"SystolicBP {latest_sys} mmHg meets WHO hypertension "
            f"in pregnancy threshold (≥140)"
        )

    # Rule 3: Severe hyperglycaemia → AMBER
    if max_bs > 11.1:
        return "AMBER", (
            f"Blood sugar {max_bs} mmol/L exceeds gestational "
            f"diabetes threshold (>11.1)"
        )

    # Rule 4: BP escalation pattern → AMBER
    if len(visits_raw) >= 2:
        first_sys = visits_raw[0].get("systolic_bp") or 0
        bp_rise   = latest_sys - first_sys
        if bp_rise >= 20:
            return "AMBER", (
                f"SystolicBP rose {bp_rise:.0f} mmHg across visits "
                f"— escalation pattern detected"
            )

    return None, None


def compute_alert_tier(
    probabilities: dict,
    patient_id: str,
    visit_importances: list,
) -> tuple[AlertTier, bool]:
    """
    Computes the alert tier for a patient.
    Returns: (alert_tier, is_suppressed)
    """
    high_risk_prob = probabilities.get("High risk", 0.0)

    # Check suppression
    if patient_id in _alert_history:
        prev = _alert_history[patient_id]
        time_since = datetime.now() - prev["last_alerted"]
        score_delta = high_risk_prob - prev["last_score"]

        if time_since < timedelta(hours=SUPPRESSION_HOURS):
            if score_delta < SUPPRESSION_DELTA:
                return prev["tier"], True

    # Determine tier
    if high_risk_prob >= RED_THRESHOLD:
        recent_weight = sum(visit_importances[-RED_CONSECUTIVE_VISITS:])
        if recent_weight > 0.4:
            tier = AlertTier.RED
        else:
            tier = AlertTier.AMBER
    elif high_risk_prob >= AMBER_THRESHOLD:
        tier = AlertTier.AMBER
    else:
        tier = AlertTier.GREEN

    # Store in history
    _alert_history[patient_id] = {
        "last_alerted": datetime.now(),
        "last_score": high_risk_prob,
        "tier": tier,
    }

    return tier, False


def generate_action_text(
    tier: AlertTier,
    top_reasons: list,
    data_quality: float,
    staff_available: Optional[int],
    blood_units: Optional[int],
) -> tuple[str, Optional[str]]:
    """Generates actionable text for the health worker and optional transfer order."""

    transfer_order = None
    reasons_text = "; ".join(top_reasons) if top_reasons else "multiple elevated indicators"

    if tier == AlertTier.GREEN:
        action = "Continue standard prenatal care schedule. No immediate action required."

    elif tier == AlertTier.AMBER:
        action = (
            f"AMBER ALERT: Elevated risk detected ({reasons_text}). "
            f"Schedule additional checkup within 72 hours. "
            f"Increase monitoring frequency to weekly."
        )
        if data_quality < 0.7:
            action += " NOTE: Data quality is low — ensure all readings are recorded at next visit."

    else:  # RED
        action = (
            f"🔴 RED ALERT: High maternal risk detected ({reasons_text}). "
            f"Refer patient to hospital immediately for full assessment."
        )

        # Resource-aware routing
        resources_ok = True
        resource_warnings = []

        if staff_available is not None and staff_available == 0:
            resources_ok = False
            resource_warnings.append("no doctors currently on duty")
        if blood_units is not None and blood_units < 2:
            resources_ok = False
            resource_warnings.append("insufficient blood supply (<2 units)")

        if not resources_ok:
            issues = " and ".join(resource_warnings)
            transfer_order = (
                f"TRANSFER ORDER: This clinic has {issues}. "
                f"Patient must be transferred to nearest equipped facility. "
                f"Call referral line: 1800-MAMA-REF. "
                f"Document: Patient at high risk of {reasons_text}."
            )
            action += f" ⚠️ Resources insufficient — transfer order generated."

        if data_quality < 0.7:
            action += " NOTE: Prediction confidence is reduced due to missing readings."

    return action, transfer_order