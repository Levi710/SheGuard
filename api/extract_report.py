# api/extract_report.py — BATTLE-HARDENED VERSION
"""
WHY THIS FILE EXISTS:
Handles every realistic image a health worker might upload:
  - Screenshot (dark or light theme, phone or computer)
  - Photo of paper (possibly skewed, shadowed, crumpled)
  - Scanned document (high resolution, clean)
  - Low-light photo (clinic with poor lighting)
  - Rotated image (phone held sideways)
  - Colored background (yellow/blue/green clinic forms)
  - Low contrast (faded ink, old paper)
  - Glare/reflection (phone flash on glossy paper)
  - Handwritten numbers on ruled paper
  - Partial image (only part of the form captured)

STRATEGY — ensemble preprocessing + ensemble OCR:
We run MULTIPLE preprocessing pipelines × MULTIPLE Tesseract configs,
score each combination by how many medically plausible values were found,
and return the best result. This is far more robust than any single pipeline.
"""

import base64
import io
import re
import pytesseract
from PIL import Image
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Tuple

router = APIRouter()

# ── Tesseract path (Windows) ──────────────────────────────────────────────────
# Change this path only if you installed Tesseract somewhere else
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# ── Schemas ───────────────────────────────────────────────────────────────────

class ExtractRequest(BaseModel):
    image_base64: str
    image_type:   str = "image/jpeg"

class ExtractedVisit(BaseModel):
    age:          Optional[float] = None
    systolic_bp:  Optional[float] = None
    diastolic_bp: Optional[float] = None
    blood_sugar:  Optional[float] = None
    body_temp:    Optional[float] = None
    heart_rate:   Optional[float] = None
    visit_date:   Optional[str]   = None

class ExtractResponse(BaseModel):
    visits:     List[ExtractedVisit]
    patient_id: Optional[str]  = None
    notes:      Optional[str]  = None
    confidence: float          = 1.0
    raw_text:   Optional[str]  = None


# ── Medical ranges ────────────────────────────────────────────────────────────
# WHY: Values outside these ranges are OCR errors, not real readings.
# We DISCARD (not clamp) out-of-range values — a clamped wrong value
# is still wrong and could mislead the risk model.
RANGES = {
    "age":          (10,   60),
    "systolic_bp":  (70,  200),
    "diastolic_bp": (40,  130),
    "blood_sugar":  (3.0, 20.0),
    "body_temp":    (35.0, 42.0),
    "heart_rate":   (40,  160),
}

# Field label aliases — every way a clinic might write each field name
ROW_ALIASES = {
    "age":          ["age", "years", "yr", "patient age", "age (yrs)", "age:", "a.g.e", "age/yrs"],
    "systolic_bp":  ["systolic", "sbp", "sys", "systolic bp", "bp sys", "bp(sys)",
                     "upper bp", "s.b.p", "s bp", "syst", "bp systolic", "bp - sys"],
    "diastolic_bp": ["diastolic", "dbp", "dia", "diastolic bp", "bp dia", "bp(dia)",
                     "lower bp", "d.b.p", "d bp", "diast", "bp diastolic", "bp - dia"],
    "blood_sugar":  ["blood sugar", "bs", "glucose", "bg", "blood glucose", "bld sugar",
                     "sugar", "b.s", "rbs", "fbs", "ppbs", "glu", "glc", "sugr"],
    "body_temp":    ["temp", "temperature", "body temp", "tmp", "fever", "t(c)",
                     "body temperature", "b.temp", "t (c)", "t(c)", "bt", "temp:"],
    "heart_rate":   ["heart rate", "hr", "pulse", "bpm", "heartrate", "h.r",
                     "heart", "p/r", "pr", "pulse rate", "hr:", "beats/min"],
}

# Lines containing these words are SKIPPED during parsing
# WHY: "Field V1 V2 V3" contains "1","2","3" which would be misread as values
SKIP_KEYWORDS = [
    "field", "visit", "v1", "v2", "v3", "v 1", "v 2", "v 3",
    "column", "header", "parameter", "reading", "measurement",
    "no.", "sl.", "s.no", "item"
]


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING PIPELINES
# Each targets a different image problem. We run several and pick the best.
# ═══════════════════════════════════════════════════════════════════════════════

def to_gray(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV grayscale array."""
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2GRAY)


def smart_upscale(gray: np.ndarray, target_width: int = 1400) -> np.ndarray:
    """
    WHY: Tesseract needs text to be at least 20-30px tall to read reliably.
    Small phone screenshots or thumbnails have tiny text that OCR misses.
    We upscale to target_width using bicubic interpolation (preserves sharp text edges).
    """
    h, w = gray.shape
    if w < target_width:
        scale = target_width / w
        gray  = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)
    return gray


def detect_and_invert_if_dark(gray: np.ndarray) -> np.ndarray:
    """
    WHY: Tesseract is trained on black text on white background.
    Dark-theme screenshots (WhatsApp dark mode, VS Code, etc.) have
    white text on dark background — Tesseract reads nothing without inversion.

    DETECTION: Mean pixel brightness < 127 = most pixels are dark = dark background.
    We bitwise-NOT the image: every dark pixel becomes bright and vice versa.
    """
    if np.mean(gray) < 127:
        return cv2.bitwise_not(gray)
    return gray


def remove_shadow(gray: np.ndarray) -> np.ndarray:
    """
    WHY: Phone photos of paper often have a shadow from the hand or phone body.
    This creates uneven brightness — one corner is darker than the rest.

    HOW (background subtraction):
    1. Dilate heavily → text pixels expand and "fill in", leaving only background
    2. Median blur → smooth the background estimate
    3. Subtract from original → removes the background brightness variation
    4. Normalise → stretch to full 0-255 range
    """
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg      = cv2.medianBlur(dilated, 21)
    diff    = 255 - cv2.absdiff(gray, bg)
    norm    = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def enhance_contrast_clahe(gray: np.ndarray) -> np.ndarray:
    """
    WHY CLAHE (Contrast Limited Adaptive Histogram Equalisation):
    Standard histogram equalisation boosts contrast globally but creates artifacts.
    CLAHE works on small image tiles separately — it enhances faded ink without
    blowing out the background, and handles images with varying contrast.

    clipLimit=2.0  — prevents over-amplifying noise in flat regions
    tileGridSize   — small enough to be local, large enough for patterns
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def deskew(gray: np.ndarray) -> np.ndarray:
    """
    WHY: Paper placed on a table and photographed is rarely perfectly flat.
    Even a 3-5 degree rotation significantly confuses Tesseract.

    HOW: Find all text pixel coordinates, fit a minimum-area bounding rectangle,
    read its angle, and rotate the image to correct the skew.
    We skip correction if the detected angle > 15 degrees (likely wrong detection).
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords    = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > 15:
        return gray   # skip unreliable angle
    (h, w) = gray.shape
    M      = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                           flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)


def remove_ruled_lines(gray: np.ndarray) -> np.ndarray:
    """
    WHY: Handwritten reports often use ruled (lined) paper.
    Horizontal lines can break through characters, confusing Tesseract's
    text segmentation. We detect and remove long horizontal line structures.

    HOW: A very wide, 1-pixel-tall kernel detects structures that are long
    horizontally = ruled lines. We subtract those from the image.
    """
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    lines   = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
    return cv2.subtract(gray, lines)


# ── 9 Preprocessing pipelines ─────────────────────────────────────────────────

def pipeline_standard(gray: np.ndarray) -> np.ndarray:
    """Pipeline A: Clean printed document, white paper, good lighting."""
    gray = smart_upscale(gray)
    gray = detect_and_invert_if_dark(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    return cv2.adaptiveThreshold(gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)


def pipeline_shadow(gray: np.ndarray) -> np.ndarray:
    """Pipeline B: Paper photo with shadow or uneven lighting."""
    gray = smart_upscale(gray)
    gray = detect_and_invert_if_dark(gray)
    gray = remove_shadow(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=8)
    return cv2.adaptiveThreshold(gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 12)


def pipeline_low_contrast(gray: np.ndarray) -> np.ndarray:
    """Pipeline C: Faded ink, old paper, poor scan quality."""
    gray = smart_upscale(gray)
    gray = detect_and_invert_if_dark(gray)
    gray = enhance_contrast_clahe(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=12)
    # Unsharp mask: enhances edges by subtracting blurred version
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    gray    = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return cv2.adaptiveThreshold(gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 8)


def pipeline_deskewed(gray: np.ndarray) -> np.ndarray:
    """Pipeline D: Rotated or skewed image (phone held sideways)."""
    gray = smart_upscale(gray)
    gray = detect_and_invert_if_dark(gray)
    gray = deskew(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    return cv2.adaptiveThreshold(gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)


def pipeline_ruled_paper(gray: np.ndarray) -> np.ndarray:
    """Pipeline E: Handwritten on ruled/lined paper."""
    gray = smart_upscale(gray)
    gray = detect_and_invert_if_dark(gray)
    gray = remove_shadow(gray)
    gray = remove_ruled_lines(gray)
    gray = enhance_contrast_clahe(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=15)
    return cv2.adaptiveThreshold(gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 8)


def pipeline_high_noise(gray: np.ndarray) -> np.ndarray:
    """Pipeline F: Low-light photo, dim room, heavily compressed JPEG."""
    gray = smart_upscale(gray, target_width=1800)
    gray = detect_and_invert_if_dark(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=20,
                                     templateWindowSize=7, searchWindowSize=21)
    gray = enhance_contrast_clahe(gray)
    # Morphological close: reconnects character strokes broken by noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gray   = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    _, out = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def pipeline_screenshot(gray: np.ndarray) -> np.ndarray:
    """
    Pipeline G: Screenshot (phone, computer, WhatsApp, EHR system).
    """
    gray = smart_upscale(gray)
    gray = detect_and_invert_if_dark(gray)
    _, out = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def pipeline_inverted(gray: np.ndarray) -> np.ndarray:
    """Pipeline H: Explicit inversion for white-on-dark text."""
    return cv2.bitwise_not(gray)


def pipeline_color_form(pil_image: Image.Image) -> np.ndarray:
    """
    Pipeline I: Colored background (yellow paper, green header, blue stripes).
    """
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan  = lab[:, :, 0]
    l_chan  = smart_upscale(l_chan)
    l_chan  = detect_and_invert_if_dark(l_chan)
    l_chan  = enhance_contrast_clahe(l_chan)
    l_chan  = cv2.fastNlMeansDenoising(l_chan, h=10)
    return cv2.adaptiveThreshold(l_chan, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)


PIPELINE_FNS = {
    "standard":     pipeline_standard,
    "shadow":       pipeline_shadow,
    "low_contrast": pipeline_low_contrast,
    "deskewed":     pipeline_deskewed,
    "ruled_paper":  pipeline_ruled_paper,
    "high_noise":   pipeline_high_noise,
    "screenshot":   pipeline_screenshot,
    "inverted":     pipeline_inverted,
}

# Pipeline order per detected image type
PIPELINE_ORDERS = {
    "screenshot":   ["screenshot", "inverted", "standard"],
    "dark_bg":      ["inverted", "screenshot", "standard"],
    "low_contrast": ["low_contrast", "shadow", "standard"],
    "colored_form": ["color_form", "low_contrast", "standard"],
    "camera_photo": ["shadow", "deskewed", "ruled_paper"],
    "standard":     ["standard", "shadow", "low_contrast"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE TYPE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def detect_image_type(gray: np.ndarray, pil_image: Image.Image) -> str:
    """Heuristically classify the image to select optimal pipeline order."""
    mean_b = float(np.mean(gray))
    std_b  = float(np.std(gray))
    h, w   = gray.shape

    if std_b > 75 and (mean_b < 70 or mean_b > 185):
        return "screenshot"

    if mean_b < 85:
        return "dark_bg"

    if std_b < 35:
        return "low_contrast"

    img_hsv  = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2HSV)
    mean_sat = float(np.mean(img_hsv[:, :, 1]))
    if mean_sat > 40:
        return "colored_form"

    if w > 2000 or h > 2000:
        return "camera_photo"

    return "standard"


# ═══════════════════════════════════════════════════════════════════════════════
# OCR ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

OCR_CONFIGS = [
    "--psm 6 --oem 3",
    "--psm 4 --oem 3",
    "--psm 11 --oem 3",
    "--psm 3 --oem 3",
]


def run_ocr_best(processed_img: np.ndarray) -> str:
    """Run Tesseract and return output with most digits."""
    pil         = Image.fromarray(processed_img)
    best_text   = ""
    best_digits = 0
    for config in OCR_CONFIGS:
        try:
            text   = pytesseract.image_to_string(pil, config=config)
            digits = sum(c.isdigit() for c in text)
            if digits > best_digits:
                best_digits = digits
                best_text   = text
        except Exception:
            continue
    return best_text


# ═══════════════════════════════════════════════════════════════════════════════
# PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

def find_numbers(text: str) -> List[float]:
    """Extract all integers and decimals from a string."""
    return [float(m) for m in re.findall(r'\d+\.?\d*', text)]


def match_label(line: str) -> Optional[str]:
    """Check if a line contains a known field name alias."""
    ll = line.lower()
    for field, aliases in ROW_ALIASES.items():
        for alias in aliases:
            if alias in ll:
                return field
    return None


def is_skip_line(line: str) -> bool:
    """Returns True if this line is a header or label-only row to skip."""
    ll = line.lower()
    return any(kw in ll for kw in SKIP_KEYWORDS)


def parse_table_format(raw_text: str) -> List[dict]:
    """Parser A -- Row-per-field table layout."""
    lines      = [l.strip() for l in raw_text.split('\n') if l.strip()]
    field_vals = {}
    max_visits = 0

    for idx, line in enumerate(lines):
        if is_skip_line(line):
            continue
        field = match_label(line)
        if field is None:
            continue

        # Robust lookahead: look for numbers in current line + next 2 lines
        search_chunk = line
        if idx + 1 < len(lines): search_chunk += " " + lines[idx + 1]
        if idx + 2 < len(lines): search_chunk += " " + lines[idx + 2]

        numbers = find_numbers(search_chunk)
        lo, hi  = RANGES[field]
        valid   = [n for n in numbers if lo <= n <= hi]
        if valid:
            if field in field_vals:
                field_vals[field].extend(valid)
            else:
                field_vals[field] = valid
            max_visits = max(max_visits, len(field_vals[field]))

    if not field_vals:
        return []

    results = []
    for i in range(max_visits):
        visit = {}
        for field in RANGES.keys():
            vals = field_vals.get(field, [])
            visit[field] = vals[i] if i < len(vals) else None
        results.append(visit)
    return results


def parse_column_format(raw_text: str) -> List[dict]:
    """Parser B -- Column-per-visit layout."""
    lines         = [l.strip() for l in raw_text.split('\n') if l.strip()]
    header_idx    = -1
    header_fields = []

    for i, line in enumerate(lines):
        # Relaxed split for mobile OCR
        parts = re.split(r'\s{1,}|\t', line)
        known = [match_label(p) for p in parts]
        if sum(f is not None for f in known) >= 2:
            header_idx    = i
            header_fields = known
            break

    if header_idx == -1:
        return []

    visits = []
    for line in lines[header_idx + 1:header_idx + 10]:
        if is_skip_line(line):
            continue
        parts = re.split(r'\s{1,}|\t', line)
        if not parts: continue
        visit = {}
        for col_idx, field in enumerate(header_fields):
            if field is None or col_idx >= len(parts):
                continue
            nums = find_numbers(parts[col_idx])
            if nums:
                lo, hi = RANGES[field]
                val    = nums[0]
                if lo <= val <= hi:
                    visit[field] = val
        if visit:
            visits.append(visit)
    return visits


def parse_key_value_format(raw_text: str) -> List[dict]:
    """Parser C -- Key-value format."""
    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
    visit = {}

    for line in lines:
        if is_skip_line(line):
            continue
        field = match_label(line)
        if field is None:
            continue

        bp_match = re.search(r'(\d{2,3})\s*/\s*(\d{2,3})', line)
        if bp_match and field in ("systolic_bp", "diastolic_bp"):
            s, d = float(bp_match.group(1)), float(bp_match.group(2))
            if 70 <= s <= 200: visit["systolic_bp"]  = s
            if 40 <= d <= 130: visit["diastolic_bp"] = d
            continue

        nums   = find_numbers(line)
        lo, hi = RANGES[field]
        valid  = [n for n in nums if lo <= n <= hi]
        if valid:
            visit[field] = valid[0]

    return [visit] if visit else []


def score_visits(visits: List[dict]) -> int:
    """Count total non-None extracted values across all visits."""
    return sum(1 for v in visits for val in v.values() if val is not None)


def best_parse(raw_text: str) -> Tuple[List[dict], str]:
    """Try all parsers and return the one that extracted the most data."""
    candidates = [
        (parse_table_format(raw_text),    "row-per-field"),
        (parse_column_format(raw_text),   "column-per-visit"),
        (parse_key_value_format(raw_text),"key-value"),
    ]
    return max(candidates, key=lambda x: score_visits(x[0]))


def extract_patient_id(raw_text: str) -> Optional[str]:
    """Find common patient ID patterns in OCR text."""
    patterns = [
        r'PT[-\s]?\d{4}[-\s]?\d+',
        r'Patient\s*(ID|No\.?|Number)[:\s]+([\w-]+)',
        r'Reg(?:istration)?\s*(No\.?|#|:)\s*([\w-]+)',
        r'ANC\s*(?:No\.?|#|:)?\s*([\w-]+)',
        r'Card\s*(?:No\.?|#)[:\s]*([\w-]+)',
        r'P/(\d+)',
        r'MRN[:\s]+([\w-]+)',
    ]
    for pattern in patterns:
        m = re.search(pattern, raw_text, re.IGNORECASE)
        if m:
            groups = [g for g in m.groups() if g]
            return groups[-1] if groups else m.group(0)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/extract-report", response_model=ExtractResponse)
async def extract_report(request: ExtractRequest):
    """POST /extract-report — Offline OCR extraction."""

    try:
        image_bytes = base64.b64decode(request.image_base64)
        pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode image: {e}"
        )

    gray      = to_gray(pil_image)
    img_type  = detect_image_type(gray, pil_image)
    pipelines = PIPELINE_ORDERS.get(img_type, PIPELINE_ORDERS["standard"])

    best_visits   = []
    best_score    = 0
    best_text     = ""
    best_pipeline = "none"
    best_parser   = "none"
    all_texts     = []

    for pipeline_name in pipelines:
        try:
            if pipeline_name == "color_form":
                processed = pipeline_color_form(pil_image)
            else:
                fn        = PIPELINE_FNS.get(pipeline_name, pipeline_standard)
                processed = fn(to_gray(pil_image))
        except Exception:
            continue

        try:
            raw_text = run_ocr_best(processed)
        except Exception:
            continue

        all_texts.append(raw_text[:200])

        visits, parser_name = best_parse(raw_text)
        score               = score_visits(visits)

        if score > best_score:
            best_score    = score
            best_visits   = visits
            best_text     = raw_text
            best_pipeline = pipeline_name
            best_parser   = parser_name

    # 4. Handle complete failure
    if not best_visits or best_score == 0:
        fallback_text = all_texts[0][:200] if all_texts else "No text extracted"
        return ExtractResponse(
            visits     = [],
            patient_id = None,
            notes      = (
                f"Image type detected: {img_type}. "
                "No structured data could be extracted. "
                "Check image quality or unusual layout."
            ),
            confidence = 0.0,
            raw_text   = fallback_text,
        )

    # 5. Build ExtractedVisit objects
    extracted = [
        ExtractedVisit(
            age          = v.get("age"),
            systolic_bp  = v.get("systolic_bp"),
            diastolic_bp = v.get("diastolic_bp"),
            blood_sugar  = v.get("blood_sugar"),
            body_temp    = v.get("body_temp"),
            heart_rate   = v.get("heart_rate"),
        )
        for v in best_visits
    ]

    total_possible = len(extracted) * 6
    total_filled   = sum(
        1 for v in extracted
        for val in [v.age, v.systolic_bp, v.diastolic_bp,
                    v.blood_sugar, v.body_temp, v.heart_rate]
        if val is not None
    )
    confidence = round(total_filled / max(total_possible, 1), 2)

    notes = (
        f"Type: {img_type} | "
        f"Pipe: {best_pipeline} | "
        f"Parser: {best_parser} | "
        f"{total_filled}/{total_possible} fields."
    )

    return ExtractResponse(
        visits     = extracted,
        patient_id = extract_patient_id(best_text),
        notes      = notes,
        confidence = confidence,
        raw_text   = best_text[:500],
    )
