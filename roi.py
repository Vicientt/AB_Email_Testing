"""
ROI simulation utilities for uplift‑based targeting.

This module contains a function to simulate incremental profit when mailing
only the top k% of customers ranked by predicted uplift.  The profit
calculation assumes a fixed margin per incremental conversion and a cost
per e‑mail sent.
"""

import numpy as np
import pandas as pd
try:
    # Allow relative import when used as part of a package
    from .uplift_model import uplift_at_k  # type: ignore
except Exception:
    # Fallback to absolute import when modules are in the same directory
    from uplift_model import uplift_at_k


def simulate_roi(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    ks: list,
    margin: float = 10.0,
    cost_email: float = 0.1,
) -> pd.DataFrame:
    """Simulate incremental profit for various mailing fractions.

    Parameters
    ----------
    y_true : ndarray
        Binary outcomes for the test set.
    treatment : ndarray
        Treatment assignments for the test set.
    uplift_scores : ndarray
        Predicted uplift scores for the test set.
    ks : list
        List of fractions (0 < k ≤ 1) to evaluate.
    margin : float, optional
        Profit per incremental conversion.
    cost_email : float, optional
        Cost per e‑mail sent.

    Returns
    -------
    pd.DataFrame
        DataFrame with k, uplift_at_k, number mailed, incremental conversions,
        revenue gain, e‑mail cost, and net profit.
    """
    n = len(y_true)
    results = []
    for k in ks:
        uplift_k = uplift_at_k(y_true, treatment, uplift_scores, k)
        top_n = int(np.floor(n * k))
        incremental_conv = uplift_k * top_n
        revenue_gain = incremental_conv * margin
        email_cost = top_n * cost_email
        net_profit = revenue_gain - email_cost
        results.append(
            {
                "k": k,
                "uplift_at_k": uplift_k,
                "n_mailed": top_n,
                "incremental_conv": incremental_conv,
                "revenue_gain": revenue_gain,
                "email_cost": email_cost,
                "net_profit": net_profit,
            }
        )
    return pd.DataFrame(results)