"""
MamaGuard — Data Pipeline
Loads UCI CSV, cleans, scales, builds sequences, and splits into train/val.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ─── Constants ────────────────────────────────────────────────────────────────

FEATURE_COLS = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
LABEL_COL    = 'RiskLevel'
LABEL_MAP    = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
SEQ_LEN      = 5


# ─── Data quality scorer ─────────────────────────────────────────────────────

def compute_data_quality(row: pd.Series) -> float:
    """Returns 0.0–1.0 indicating how complete and plausible the data is."""
    score, total = 0.0, 0

    for col in FEATURE_COLS:
        total += 1
        if pd.notna(row.get(col)):
            score += 1.0

    plausibility = {
        'Age': (10, 60), 'SystolicBP': (70, 200), 'DiastolicBP': (40, 130),
        'BS': (3.0, 20.0), 'BodyTemp': (35.0, 42.0), 'HeartRate': (40, 160),
    }
    for col, (lo, hi) in plausibility.items():
        total += 1
        val = row.get(col)
        if pd.notna(val) and lo <= val <= hi:
            score += 1.0

    return round(score / total, 3)


# ─── Sequence builders ────────────────────────────────────────────────────────

def build_type_a_sequences(X: np.ndarray, y: np.ndarray,
                            quality: np.ndarray) -> tuple:
    """Type A — Within-risk-group sliding window sequences."""
    X_seq, y_seq, q_seq = [], [], []

    for risk_label in [0, 1, 2]:
        idx = np.where(y == risk_label)[0]
        if len(idx) < SEQ_LEN:
            continue

        age_order  = np.argsort(X[idx, 0])
        idx_sorted = idx[age_order]

        for i in range(len(idx_sorted) - SEQ_LEN + 1):
            seq_idx = idx_sorted[i : i + SEQ_LEN]
            X_seq.append(X[seq_idx].copy())
            y_seq.append(risk_label)
            q_seq.append(quality[seq_idx].mean())

    return X_seq, y_seq, q_seq


def build_type_b_sequences(X: np.ndarray, y: np.ndarray,
                            rng: np.random.Generator) -> tuple:
    """Type B — Synthetic escalation sequences for mid and high risk."""
    X_seq, y_seq, q_seq = [], [], []

    for risk_label in [1, 2]:
        indices = np.where(y == risk_label)[0]
        for idx in indices:
            anchor = X[idx]
            seq = []
            for step in range(SEQ_LEN):
                fraction = 0.30 + 0.70 * (step / (SEQ_LEN - 1))
                noise = rng.normal(0, 0.05, anchor.shape)
                seq.append(anchor * fraction + noise)
            X_seq.append(np.array(seq, dtype=np.float32))
            y_seq.append(risk_label)
            q_seq.append(1.0)

    return X_seq, y_seq, q_seq


def build_type_c_sequences(X: np.ndarray, y: np.ndarray,
                            rng: np.random.Generator,
                            n_per_transition: int = 80) -> tuple:
    """Type C — Cross-risk deterioration sequences (low->mid, low->high, mid->high)."""
    X_seq, y_seq, q_seq = [], [], []

    low_idx  = np.where(y == 0)[0]
    mid_idx  = np.where(y == 1)[0]
    high_idx = np.where(y == 2)[0]

    if len(low_idx) == 0 or len(mid_idx) == 0 or len(high_idx) == 0:
        return X_seq, y_seq, q_seq

    def blend(a: np.ndarray, b: np.ndarray, frac: float = 0.5) -> np.ndarray:
        """Linear interpolation between two rows + small noise."""
        return a * (1 - frac) + b * frac + rng.normal(0, 0.05, a.shape)

    def pick(indices: np.ndarray) -> np.ndarray:
        """Pick one random row from an index array."""
        return X[rng.choice(indices)]

    # Transition 1: LOW -> HIGH
    for _ in range(n_per_transition):
        seq = [
            pick(low_idx), pick(low_idx),
            blend(pick(low_idx), pick(high_idx)),
            pick(high_idx), pick(high_idx),
        ]
        X_seq.append(np.array(seq, dtype=np.float32))
        y_seq.append(2)
        q_seq.append(1.0)

    # Transition 2: LOW -> MID
    for _ in range(n_per_transition):
        seq = [
            pick(low_idx), pick(low_idx), pick(low_idx),
            pick(mid_idx), pick(mid_idx),
        ]
        X_seq.append(np.array(seq, dtype=np.float32))
        y_seq.append(1)
        q_seq.append(1.0)

    # Transition 3: MID -> HIGH
    for _ in range(n_per_transition):
        seq = [
            pick(mid_idx), pick(mid_idx),
            blend(pick(mid_idx), pick(high_idx)),
            pick(high_idx), pick(high_idx),
        ]
        X_seq.append(np.array(seq, dtype=np.float32))
        y_seq.append(2)
        q_seq.append(1.0)

    return X_seq, y_seq, q_seq


# ─── Internal helper ─────────────────────────────────────────────────────────

def _build_all_sequences(X: np.ndarray, y: np.ndarray,
                          quality: np.ndarray,
                          rng: np.random.Generator,
                          include_augmentation: bool,
                          label: str = "") -> tuple:
    """Builds all sequence types from a given set of scaled rows."""
    X_all, y_all, q_all = [], [], []

    a_X, a_y, a_q = build_type_a_sequences(X, y, quality)
    X_all += a_X; y_all += a_y; q_all += a_q
    print(f"  {label} Type A (within-class):  {len(a_X):5d} sequences")

    if include_augmentation:
        b_X, b_y, b_q = build_type_b_sequences(X, y, rng)
        X_all += b_X; y_all += b_y; q_all += b_q
        print(f"  {label} Type B (escalation):    {len(b_X):5d} sequences")

        min_cls = min(sum(y == 0), sum(y == 1), sum(y == 2))
        if min_cls >= SEQ_LEN:
            c_X, c_y, c_q = build_type_c_sequences(X, y, rng)
            X_all += c_X; y_all += c_y; q_all += c_q
            print(f"  {label} Type C (cross-risk):    {len(c_X):5d} sequences")
        else:
            print(f"  {label} Type C skipped (smallest class has {min_cls} rows)")

    return X_all, y_all, q_all


# ─── Main pipeline ────────────────────────────────────────────────────────────

def load_and_preprocess(csv_path: str):
    """
    Returns pre-split, pre-scaled sequence arrays + the fitted scaler.

    Returns:
        X_train, y_train, q_train, X_val, y_val, q_val, scaler
    """

    # 1. Load
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    df[LABEL_COL] = df[LABEL_COL].str.lower().str.strip()

    # 2. Quality scores BEFORE imputation
    quality_per_row = df.apply(compute_data_quality, axis=1).values

    # 3. Impute missing values
    for col in FEATURE_COLS:
        df[col] = df[col].fillna(df[col].median())

    X_raw = df[FEATURE_COLS].values.astype(np.float32)
    y_raw = df[LABEL_COL].map(LABEL_MAP).values

    # 4. Split raw rows first (prevents data leakage and sequence overlap)
    (X_train_raw, X_val_raw,
     y_train_raw, y_val_raw,
     q_train_raw, q_val_raw) = train_test_split(
        X_raw, y_raw, quality_per_row,
        test_size=0.2, random_state=42, stratify=y_raw
    )
    print(f"Raw split -> train: {len(X_train_raw)} rows | val: {len(X_val_raw)} rows")
    print(f"Train classes: low={sum(y_train_raw==0)} "
          f"mid={sum(y_train_raw==1)} high={sum(y_train_raw==2)}")
    print(f"Val   classes: low={sum(y_val_raw==0)} "
          f"mid={sum(y_val_raw==1)} high={sum(y_val_raw==2)}")

    # 5. Fit scaler on train rows only
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled   = scaler.transform(X_val_raw)

    # 6. Build sequences separately
    rng = np.random.default_rng(42)

    print("\nBuilding TRAINING sequences:")
    train_seqs = _build_all_sequences(
        X_train_scaled, y_train_raw, q_train_raw,
        rng, include_augmentation=True, label="Train"
    )

    print("\nBuilding VALIDATION sequences:")
    val_seqs = _build_all_sequences(
        X_val_scaled, y_val_raw, q_val_raw,
        rng, include_augmentation=False, label="Val"
    )

    X_train, y_train, q_train = [np.array(a) for a in train_seqs]
    X_val,   y_val,   q_val   = [np.array(a) for a in val_seqs]

    print(f"\nFinal -> Train: {len(X_train)} seqs | Val: {len(X_val)} seqs")
    print(f"Train dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Val   dist: {dict(zip(*np.unique(y_val,   return_counts=True)))}")

    return (X_train.astype(np.float32), y_train.astype(np.int64),
            q_train.astype(np.float32),
            X_val.astype(np.float32),   y_val.astype(np.int64),
            q_val.astype(np.float32),
            scaler)


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class MaternalDataset(Dataset):
    def __init__(self, X, y, quality):
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.long)
        self.q = torch.tensor(np.array(quality), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.q[idx]


def get_dataloaders(csv_path: str, batch_size: int = 32):
    """Returns train/val DataLoaders and the fitted scaler."""
    (X_train, y_train, q_train,
     X_val,   y_val,   q_val,
     scaler) = load_and_preprocess(csv_path)

    train_ds = MaternalDataset(X_train, y_train, q_train)
    val_ds   = MaternalDataset(X_val,   y_val,   q_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print(f"\nDataLoaders ready:")
    print(f"  Train: {len(train_ds)} sequences | "
          f"{len(train_loader)} batches of {batch_size}")
    print(f"  Val:   {len(val_ds)} sequences | "
          f"{len(val_loader)} batches")

    return train_loader, val_loader, scaler