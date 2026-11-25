"""
Trains and evaluates flare-probability models.

Models:
    - Logistic Regression (balanced)
    - Random Forest (balanced)
    - Persistence baseline: per-hour probability = fraction of prior 24h with a flare

Split: chronological with a positivity guard. Walk the boundary back until the
test window has at least MIN_TEST_POS M/X positives. Refuse to train otherwise.

Metrics:
    - Brier score (lower better)
    - Calibration (reliability curve + ECE)
    - TSS = TPR - FPR (higher better)

Run:
    python -m src.train_model
Outputs:
    data/model.pkl, data/metrics.json, calibration_*.png, feature_importance.png
"""
import json
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, roc_curve

FEATURES = [
    "flux_log_mean", "flux_log_max", "flux_log_min", "flux_log_std",
    "flux_mean_1h", "flux_mean_3h", "flux_mean_6h", "flux_mean_12h", "flux_mean_24h",
    "flux_max_1h", "flux_max_3h", "flux_max_6h", "flux_max_24h",
    "flux_delta_1h", "flux_delta_3h",
    "hour_sin", "hour_cos",
]
LABEL = "flare_next_24h"
MIN_TEST_POS = 3


def tss(y_true, y_pred):
    """TPR - FPR at optimal threshold. nan if no positives."""
    if y_true.sum() == 0:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    j = tpr - fpr
    return float(j[np.argmax(j)])


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def reliability_plot(y_true, y_prob, path, title):
    bins = np.linspace(0, 1, 11)
    centers, obs = [], []
    for i in range(10):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            centers.append((bins[i] + bins[i + 1]) / 2)
            obs.append(y_true[mask].mean())
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    if centers:
        plt.plot(centers, obs, "o-", color="crimson", label="model")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def choose_split(df):
    """
    Start at 80/20. Walk the boundary back in 5% steps until the test window
    has at least MIN_TEST_POS positives. Cap at 60/40.
    """
    n = len(df)
    y = df[LABEL].values
    for frac in np.arange(0.80, 0.59, -0.05):
        split = int(n * frac)
        test_pos = int(y[split:].sum())
        if test_pos >= MIN_TEST_POS:
            return split
    split = int(n * 0.60)
    test_pos = int(y[split:].sum())
    if test_pos < 2:
        raise SystemExit(
            f"Refusing to train: test window has only {test_pos} M/X positives "
            f"even at 60/40. Dataset too small. Widen the flux window first."
        )
    return split


def persistence_baseline(df, split_idx):
    """
    For each test hour, prob = fraction of prior 24h rows (all available, including
    train tail) with label == 1. Leak-free: uses only past labels.
    """
    y = df[LABEL].values
    probs = np.full(len(df) - split_idx, np.nan)
    for i, idx in enumerate(range(split_idx, len(df))):
        lo = max(0, idx - 24)
        window = y[lo:idx]
        probs[i] = float(window.mean()) if len(window) > 0 else 0.0
    return probs


def main():
    os.makedirs("data", exist_ok=True)
    df = pd.read_parquet("data/training.parquet").sort_values("hour").reset_index(drop=True)
    if df.empty:
        raise SystemExit("training.parquet is empty — run build_dataset first")

    df = df.dropna(subset=FEATURES + [LABEL]).reset_index(drop=True)

    split = choose_split(df)
    X, y = df[FEATURES].values, df[LABEL].values
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    print(f"train: {len(X_tr)} rows ({y_tr.mean()*100:.2f}% pos)")
    print(f"test:  {len(X_te)} rows ({y_te.mean()*100:.2f}% pos, {int(y_te.sum())} positives)")

    results = {}

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    lr.fit(X_tr, y_tr)
    p_lr = lr.predict_proba(X_te)[:, 1]
    results["logistic_regression"] = {
        "brier": float(brier_score_loss(y_te, p_lr)),
        "tss": tss(y_te, p_lr),
        "ece": expected_calibration_error(y_te, p_lr),
    }
    reliability_plot(y_te, p_lr, "data/calibration_lr.png", "Logistic Regression")

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    rf.fit(X_tr, y_tr)
    p_rf = rf.predict_proba(X_te)[:, 1]
    results["random_forest"] = {
        "brier": float(brier_score_loss(y_te, p_rf)),
        "tss": tss(y_te, p_rf),
        "ece": expected_calibration_error(y_te, p_rf),
    }
    reliability_plot(y_te, p_rf, "data/calibration_rf.png", "Random Forest")

    p_base = persistence_baseline(df, split)
    results["persistence_baseline"] = {
        "brier": float(brier_score_loss(y_te, p_base)),
        "tss": tss(y_te, p_base),
        "ece": expected_calibration_error(y_te, p_base),
    }

    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    imp.head(12).plot(kind="barh", color="steelblue")
    plt.gca().invert_yaxis()
    plt.title("Random Forest feature importance")
    plt.tight_layout()
    plt.savefig("data/feature_importance.png", dpi=120)
    plt.close()

    print("\n=== metrics ===")
    for name, m in results.items():
        print(f"{name:24s}  brier={m['brier']:.4f}  tss={m['tss']:.3f}  ece={m['ece']:.4f}")

    winner = min(results, key=lambda k: results[k]["brier"])
    print(f"\nwinner (by brier): {winner}")

    joblib.dump({"lr": lr, "rf": rf, "winner": winner, "features": FEATURES}, "data/model.pkl")
    with open("data/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved: data/model.pkl  data/metrics.json  data/calibration_*.png  data/feature_importance.png")


if __name__ == "__main__":
    main()