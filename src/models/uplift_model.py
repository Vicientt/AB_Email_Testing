"""
Uplift modeling utilities using a T-learner with Random Forests.

This module implements:
- Feature preprocessing (numeric + categorical via ColumnTransformer)
- T-learner Random Forest with probability calibration
- Qini curve computation
- Qini AUC calculation
- uplift@k evaluation
- Qini plotting utilities

Used in the main pipeline for the Hillstrom uplift experiment.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. FEATURE PREPROCESSING
# ----------------------------------------------------------------------
def preprocess_features(df: pd.DataFrame, numeric_cols: list, categorical_cols: list):
    """Create a ColumnTransformer for mixed feature types."""
    transformer = ColumnTransformer(
        [
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return transformer


# ----------------------------------------------------------------------
# 2. TRAIN T-LEARNER
# ----------------------------------------------------------------------
def train_uplift_tlearner(
    df: pd.DataFrame,
    features: list,
    test_size: float = 0.3,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = 6,
) -> pd.DataFrame:
    """
    Train uplift model using T-learner:
    - train RF on treated subset
    - train RF on control subset
    - predict P(y|treat) - P(y|control)
    """
    X = df[features]
    y = df["conversion"].astype(int)
    t = df["treatment"].astype(int)

    # Stratify by outcome + treatment → balanced split
    strata = (t.astype(str) + "_" + y.astype(str))

    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, t,
        test_size=test_size,
        random_state=random_state,
        stratify=strata
    )

    # Auto-detect types
    numeric_cols = [col for col in features if df[col].dtype != object]
    categorical_cols = [col for col in features if df[col].dtype == object]

    # Transform features
    transformer = preprocess_features(df, numeric_cols, categorical_cols)
    X_train_trans = transformer.fit_transform(X_train)
    X_test_trans = transformer.transform(X_test)

    # Base models
    base_t = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    base_c = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )

    # Probability calibration
    model_t = CalibratedClassifierCV(base_t, method="isotonic", cv=3)
    model_c = CalibratedClassifierCV(base_c, method="isotonic", cv=3)

    # Fit models separately
    model_t.fit(X_train_trans[t_train == 1], y_train[t_train == 1])
    model_c.fit(X_train_trans[t_train == 0], y_train[t_train == 0])

    # Predict probabilities
    prob_t = model_t.predict_proba(X_test_trans)[:, 1]
    prob_c = model_c.predict_proba(X_test_trans)[:, 1]

    uplift = prob_t - prob_c

    return pd.DataFrame(
        {
            "y_true": y_test.values,
            "treatment": t_test.values,
            "uplift_pred": uplift,
        }
    )


# ----------------------------------------------------------------------
# 3. QINI CURVE
# ----------------------------------------------------------------------
def qini_curve(y_true: np.ndarray, uplift_scores: np.ndarray, treatment: np.ndarray):
    """
    Compute Qini curve:
    - Sort by uplift scores
    - Compute cumulative uplift difference
    """
    n = len(y_true)
    order = np.argsort(-uplift_scores)

    y_sorted = y_true[order]
    t_sorted = treatment[order]

    N_t = t_sorted.sum()
    N_c = n - N_t

    # cumulative number of responders
    cum_t_y1 = np.cumsum(y_sorted * t_sorted)
    cum_c_y1 = np.cumsum(y_sorted * (1 - t_sorted))

    phi = np.arange(1, n + 1) / n  # fraction of population
    qini = (cum_t_y1 / N_t) - (cum_c_y1 / N_c)

    return phi, qini


# ----------------------------------------------------------------------
# 4. QINI AUC
# ----------------------------------------------------------------------
def qini_auc(phi: np.ndarray, qini: np.ndarray, treatment: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute normalized Qini coefficient.
    """
    area_model = np.trapz(qini, phi)

    N_t = treatment.sum()
    N_c = len(treatment) - N_t

    delta = (y_true[treatment == 1].sum() / N_t) - (y_true[treatment == 0].sum() / N_c)
    area_random = delta / 2.0
    area_perfect = delta

    if area_perfect - area_random == 0:
        return 0.0

    return (area_model - area_random) / (area_perfect - area_random)


# ----------------------------------------------------------------------
# 5. UPLIFT @ K
# ----------------------------------------------------------------------
def uplift_at_k(y_true: np.ndarray, treatment: np.ndarray, uplift_scores: np.ndarray, k: float) -> float:
    """
    Compute uplift@k for the top k fraction of individuals.
    """
    assert 0 < k <= 1
    n = len(y_true)
    top_n = int(np.floor(n * k))
    if top_n == 0:
        return 0.0

    order = np.argsort(-uplift_scores)
    idx = order[:top_n]

    y_top = y_true[idx]
    t_top = treatment[idx]

    mask_t = t_top == 1
    mask_c = t_top == 0

    if mask_t.sum() == 0 or mask_c.sum() == 0:
        return 0.0

    return y_top[mask_t].mean() - y_top[mask_c].mean()


# ----------------------------------------------------------------------
# 6. PLOT QINI
# ----------------------------------------------------------------------
def plot_qini_vs_k(y_true, uplift_scores, treatment, title=None, save_path=None, show=False):
    """
    Plot Qini curve vs % population targeted.
    """
    y_true = np.asarray(y_true)
    uplift_scores = np.asarray(uplift_scores)
    treatment = np.asarray(treatment)

    # Qini curve
    phi, qini = qini_curve(y_true, uplift_scores, treatment)
    k_pct = phi * 100.0

    # Random baseline
    p_t = y_true[treatment == 1].mean()
    p_c = y_true[treatment == 0].mean()
    delta = p_t - p_c
    qini_random = delta * phi

    plt.figure()
    plt.plot(k_pct, qini, label="Model")
    plt.plot(k_pct, qini_random, "--", label="Random baseline")
    plt.xlabel("k (% of customers)")
    plt.ylabel("Qini (cumulative uplift)")
    if title:
        plt.title(title)
    plt.legend()

    # Save file
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()
