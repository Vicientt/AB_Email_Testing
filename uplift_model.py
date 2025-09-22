"""
Uplift modeling utilities using a T‑learner approach with random forests.

This module implements functions to train separate RandomForestClassifiers on
treatment and control data, compute predicted uplift, derive Qini curves,
calculate Qini AUC, evaluate uplift@k, and rank customers for targeting.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV


def preprocess_features(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """Create a ColumnTransformer for mixed numeric and categorical features.

    Numeric columns are passed through unchanged, and categorical columns are
    one‑hot encoded.  Unknown categories at prediction time are ignored.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset (used only to infer column types).
    numeric_cols : list
        List of numerical feature names.
    categorical_cols : list
        List of categorical feature names.

    Returns
    -------
    ColumnTransformer
        A transformer suitable for fitting in a pipeline.
    """
    transformer = ColumnTransformer(
        [
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return transformer


def train_uplift_tlearner(
    df: pd.DataFrame,
    features: list,
    test_size: float = 0.3,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = 6,
) -> pd.DataFrame:
    """Train a T‑learner uplift model using RandomForestClassifiers.

    A T‑learner fits two separate models: one on treatment observations and
    one on control observations.  The uplift score for an individual is
    estimated as the difference between the predicted conversion probability
    under treatment and under control.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataset with ``conversion`` and ``treatment`` columns.
    features : list
        List of feature column names to use for modeling.
    test_size : float, optional
        Proportion of the dataset to include in the hold‑out test split.
    random_state : int, optional
        Random state for reproducibility.
    n_estimators : int, optional
        Number of trees in each random forest.
    max_depth : int, optional
        Maximum tree depth for each random forest.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the true outcomes, treatment indicators and
        predicted uplift scores for the test set.
    """
    X = df[features]
    y = df["conversion"].astype(int)
    t = df["treatment"].astype(int)
    strata = (t.astype(str) + "_" + y.astype(str)) # stratify theo cả treatment & outcome
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, t, test_size=test_size, random_state=random_state, stratify=strata
    )
    numeric_cols = [col for col in features if df[col].dtype != object]
    categorical_cols = [col for col in features if df[col].dtype == object]
    transformer = preprocess_features(df, numeric_cols, categorical_cols)
    X_train_trans = transformer.fit_transform(X_train)
    X_test_trans = transformer.transform(X_test)

    # mô hình cơ sở
    base_t = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    base_c = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    # calibration
    model_t = CalibratedClassifierCV(base_t, method="isotonic", cv=3)
    model_c = CalibratedClassifierCV(base_c, method="isotonic", cv=3)

    model_t.fit(X_train_trans[t_train == 1], y_train[t_train == 1])
    model_c.fit(X_train_trans[t_train == 0], y_train[t_train == 0])
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


def qini_curve(y_true: np.ndarray, uplift_scores: np.ndarray, treatment: np.ndarray):
    """Compute Qini curve coordinates.

    The Qini curve represents the difference in cumulative response rates
    between treated and control groups as the population is ordered by
    predicted uplift.  See [Radcliffe, 2007] for details.

    Parameters
    ----------
    y_true : array-like
        Binary outcome indicators.
    uplift_scores : array-like
        Predicted uplift scores.
    treatment : array-like
        Binary treatment assignment.

    Returns
    -------
    tuple of ndarray
        Fractions of population targeted and corresponding Qini values.
    """
    n = len(y_true)
    order = np.argsort(-uplift_scores)
    y_sorted = y_true[order]
    t_sorted = treatment[order]
    N_t = t_sorted.sum()
    N_c = n - N_t
    cum_t_y1 = np.cumsum(y_sorted * t_sorted)
    cum_c_y1 = np.cumsum(y_sorted * (1 - t_sorted))
    phi = np.arange(1, n + 1) / n
    qini = (cum_t_y1 / N_t) - (cum_c_y1 / N_c)
    return phi, qini


def qini_auc(phi: np.ndarray, qini: np.ndarray, treatment: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate the normalized area under the Qini curve.

    The Qini coefficient is computed by subtracting the area under the
    random chance line and dividing by the theoretical maximum area.  A
    value close to 1 indicates a strong uplift model.

    Parameters
    ----------
    phi : ndarray
        Fractions of the population targeted.
    qini : ndarray
        Qini values corresponding to ``phi``.
    treatment : ndarray
        Treatment indicators for the test set.
    y_true : ndarray
        True outcomes for the test set.

    Returns
    -------
    float
        The normalized Qini area (Qini coefficient).
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


def uplift_at_k(y_true: np.ndarray, treatment: np.ndarray, uplift_scores: np.ndarray, k: float) -> float:
    """Compute uplift@k for the top k fraction of customers.

    Uplift@k is defined as the difference in response rates between
    treatment and control among the top k% individuals ranked by
    predicted uplift.  If either subgroup is absent, the uplift is zero.

    Parameters
    ----------
    y_true : ndarray
        Binary outcomes.
    treatment : ndarray
        Treatment indicators.
    uplift_scores : ndarray
        Predicted uplift scores.
    k : float
        Fraction of the population to evaluate (0 < k ≤ 1).

    Returns
    -------
    float
        Estimated uplift for the top k fraction.
    """
    assert 0 < k <= 1, "k must be within (0,1]"
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