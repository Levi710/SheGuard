/* ══════════════════════════════════════════════════════════════════════════
   MamaGuard — Dashboard Application Logic
   ══════════════════════════════════════════════════════════════════════════ */

const API_BASE = "http://localhost:8000";

/* ── State ─────────────────────────────────────────────────────────────── */

let visitCount = 0;
let currentImageBase64 = null;
let currentImageType = null;


/* ── Visit Management ──────────────────────────────────────────────────── */

function addVisit(prefill = {}) {
  visitCount++;
  const container = document.getElementById("visits-container");
  const block = document.createElement("div");
  block.className = "visit-block";
  block.id = `visit-${visitCount}`;

  const v = visitCount;
  block.innerHTML = `
    <h4>Visit ${v}
      ${prefill.visit_date
        ? `<span style="font-weight:normal;color:var(--text-muted);font-size:0.78rem;margin-left:8px;">${prefill.visit_date}</span>`
        : ""}
    </h4>
    <div class="grid-3">
      <div>
        <label>Age *</label>
        <input type="number" id="v${v}_age" placeholder="e.g. 28" min="10" max="60">
      </div>
      <div>
        <label>Systolic BP * (mmHg)</label>
        <input type="number" id="v${v}_sbp" placeholder="e.g. 118" min="70" max="200">
      </div>
      <div>
        <label>Diastolic BP * (mmHg)</label>
        <input type="number" id="v${v}_dbp" placeholder="e.g. 75" min="40" max="130">
      </div>
      <div>
        <label>Blood sugar (mmol/L)</label>
        <input type="number" id="v${v}_bs" placeholder="e.g. 7.2" min="3" max="20" step="0.1">
      </div>
      <div>
        <label>Body temp (&deg;C)</label>
        <input type="number" id="v${v}_bt" placeholder="e.g. 36.8" min="35" max="42" step="0.1">
      </div>
      <div>
        <label>Heart rate (bpm)</label>
        <input type="number" id="v${v}_hr" placeholder="e.g. 78" min="40" max="160">
      </div>
    </div>`;
  container.appendChild(block);

  if (Object.keys(prefill).length > 0) {
    fillVisit(v, prefill);
  }
}

function fillVisit(v, data) {
  const fieldMap = {
    age:          `v${v}_age`,
    systolic_bp:  `v${v}_sbp`,
    diastolic_bp: `v${v}_dbp`,
    blood_sugar:  `v${v}_bs`,
    body_temp:    `v${v}_bt`,
    heart_rate:   `v${v}_hr`,
  };

  for (const [key, inputId] of Object.entries(fieldMap)) {
    const val = data[key];
    const el = document.getElementById(inputId);
    if (el && val !== null && val !== undefined) {
      el.value = val;
      el.classList.add("autofilled");
      el.addEventListener("input", () => el.classList.remove("autofilled"), { once: true });
    }
  }
}


/* ── File Upload ───────────────────────────────────────────────────────── */

function handleFileSelect(input) {
  const file = input.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    const dataUrl = e.target.result;
    const [header, base64] = dataUrl.split(",");
    currentImageBase64 = base64;
    currentImageType = file.type || "image/jpeg";

    document.getElementById("preview-img").src = dataUrl;
    document.getElementById("preview-wrap").style.display = "block";
    document.getElementById("btn-extract").style.display = "block";
    document.getElementById("extract-status").className = "extract-status";
    document.getElementById("extract-status").textContent = "";
  };
  reader.readAsDataURL(file);
}

function clearUpload() {
  currentImageBase64 = null;
  currentImageType = null;
  document.getElementById("file-input").value = "";
  document.getElementById("preview-wrap").style.display = "none";
  document.getElementById("btn-extract").style.display = "none";
  document.getElementById("extract-status").className = "extract-status";
  document.getElementById("extract-status").textContent = "";
  document.getElementById("confidence-strip").style.display = "none";
}


/* ── Drag & Drop ───────────────────────────────────────────────────────── */

document.addEventListener("DOMContentLoaded", () => {
  const zone = document.getElementById("upload-zone");

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });

  zone.addEventListener("dragleave", () => {
    zone.classList.remove("drag-over");
  });

  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) {
      const dt = new DataTransfer();
      dt.items.add(file);
      document.getElementById("file-input").files = dt.files;
      handleFileSelect(document.getElementById("file-input"));
    }
  });

  // Add first visit on load
  addVisit();
});


/* ── Image Extraction ──────────────────────────────────────────────────── */

async function extractFromImage() {
  if (!currentImageBase64) return;

  const statusEl = document.getElementById("extract-status");
  const extractBtn = document.getElementById("btn-extract");

  statusEl.className = "extract-status loading";
  statusEl.textContent = "Reading the report... (2-5 seconds)";
  extractBtn.disabled = true;

  try {
    const resp = await fetch(`${API_BASE}/extract-report`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_base64: currentImageBase64,
        image_type: currentImageType,
      }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || `Server error ${resp.status}`);
    }

    const data = await resp.json();

    if (data.patient_id) {
      const pidField = document.getElementById("patient_id");
      if (!pidField.value) {
        pidField.value = data.patient_id;
        pidField.classList.add("autofilled");
      }
    }

    document.getElementById("visits-container").innerHTML = "";
    visitCount = 0;

    const visits = data.visits || [];
    if (visits.length === 0) {
      throw new Error("No visit data found in the image. Try a clearer photo.");
    }

    visits.forEach(visit => addVisit(visit));

    const confPct = Math.round((data.confidence || 1.0) * 100);
    const confColor = confPct >= 80 ? "var(--green)" : confPct >= 60 ? "var(--amber)" : "var(--red)";
    const confEl = document.getElementById("confidence-strip");
    confEl.style.display = "block";
    confEl.innerHTML = `
      <span style="color:${confColor};font-weight:600;">
        Extraction confidence: ${confPct}%
      </span>
      ${confPct < 80 ? " -- please verify highlighted fields before submitting" : " -- all fields extracted successfully"}
      ${data.notes ? `<br><span style="color:var(--text-muted);">${data.notes}</span>` : ""}
    `;

    statusEl.className = "extract-status success";
    statusEl.textContent = `Extracted ${visits.length} visit${visits.length > 1 ? "s" : ""} -- highlighted fields are auto-filled. Verify before submitting.`;

  } catch (err) {
    statusEl.className = "extract-status error";
    statusEl.textContent = `Extraction failed: ${err.message}`;
  } finally {
    extractBtn.disabled = false;
  }
}


/* ── Prediction Submission ─────────────────────────────────────────────── */

function getVal(id) {
  const v = document.getElementById(id)?.value;
  return (v === "" || v === null || v === undefined) ? null : parseFloat(v);
}

async function submitPrediction() {
  document.getElementById("result-card").style.display = "none";
  document.getElementById("error-box").style.display = "none";
  document.getElementById("loader").style.display = "block";

  const patientId = document.getElementById("patient_id").value.trim();
  if (!patientId) { showError("Please enter a Patient ID"); return; }

  const visits = [];
  for (let i = 1; i <= visitCount; i++) {
    const age = getVal(`v${i}_age`);
    const sbp = getVal(`v${i}_sbp`);
    const dbp = getVal(`v${i}_dbp`);
    if (age === null || sbp === null || dbp === null) continue;
    visits.push({
      age, systolic_bp: sbp, diastolic_bp: dbp,
      blood_sugar: getVal(`v${i}_bs`),
      body_temp:   getVal(`v${i}_bt`),
      heart_rate:  getVal(`v${i}_hr`),
    });
  }

  if (visits.length === 0) {
    showError("Please fill in at least one visit (Age, Systolic BP, Diastolic BP are required)");
    return;
  }

  const payload = {
    patient_id: patientId,
    visits,
    staff_available: getVal("staff_available"),
    blood_units:     getVal("blood_units"),
  };

  try {
    const resp = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || `Server error ${resp.status}`);
    }
    showResult(await resp.json());
  } catch (err) {
    showError(`Could not reach the server: ${err.message}. Is the API running?`);
  } finally {
    document.getElementById("loader").style.display = "none";
  }
}


/* ── Result Display ────────────────────────────────────────────────────── */

function showResult(data) {
  const tierClass  = { GREEN: "alert-green", AMBER: "alert-amber", RED: "alert-red" };
  const badgeClass = { GREEN: "badge-green", AMBER: "badge-amber", RED: "badge-red" };
  const tierLabel  = {
    GREEN: "Low risk",
    AMBER: "Medium risk -- monitor closely",
    RED:   "High risk -- refer immediately"
  };

  const card = document.getElementById("result-card");
  const tier = data.alert_tier;

  card.className = `card result ${tierClass[tier]}`;
  card.style.display = "block";

  document.getElementById("risk-badge").textContent = tierLabel[tier];
  document.getElementById("risk-badge").className = `risk-badge ${badgeClass[tier]}`;

  document.getElementById("confidence-text").textContent =
    `Model confidence: ${(data.confidence * 100).toFixed(1)}%` +
    (data.suppressed ? " (alert suppressed -- already sent within 48h)" : "");

  const qPct = Math.round(data.data_quality * 100);
  document.getElementById("quality-bar").style.width = `${qPct}%`;
  document.getElementById("quality-bar").style.background = qPct >= 70 ? "var(--green)" : "var(--amber)";
  document.getElementById("quality-text").textContent =
    `Data quality: ${qPct}%` + (qPct < 70 ? " -- low confidence, collect missing readings" : " -- good");

  document.getElementById("reasons-list").innerHTML =
    data.top_reasons.map(r => `<div class="reason-item">${r}</div>`).join("");

  document.getElementById("action-text").textContent = data.action_required;

  const transferBox = document.getElementById("transfer-box");
  if (data.transfer_order) {
    document.getElementById("transfer-text").textContent = data.transfer_order;
    transferBox.style.display = "block";
  } else {
    transferBox.style.display = "none";
  }

  card.scrollIntoView({ behavior: "smooth" });
}

function showError(msg) {
  document.getElementById("loader").style.display = "none";
  const box = document.getElementById("error-box");
  box.textContent = msg;
  box.style.display = "block";
}
