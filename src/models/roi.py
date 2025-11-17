"""
ROI simulation utilities for uplift-based targeting.

This module contains tools for simulating incremental profit when emailing only
the top k% of customers ranked by predicted uplift scores.

The profit calculation follows:

    incremental_conversions = uplift_at_k * n_targeted
    revenue_gain = incremental_conversions * margin
    email_cost   = n_targeted * cost_email
    net_profit   = revenue_gain - email_cost

Used inside the main pipeline for evaluating uplift models on the Hillstrom dataset.
"""

import numpy as np
import pandas as pd

# Try relative import first (when used as part of package)
try:
    from .uplift_model import uplift_at_k
except Exception:
    # Fallback for standalone execution
    from uplift_model import uplift_at_k


def simulate_roi(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    ks: list,
    margin: float = 10.0,
    cost_email: float = 0.1,
) -> pd.DataFrame:
    """
    Simulate profit at various targeting percentages k.

    Parameters
    ----------
    y_true : np.ndarray
        1/0 binary response variable (conversion outcome).
    treatment : np.ndarray
        Treatment assignment (1 for treated, 0 for control).
    uplift_scores : np.ndarray
        Predicted uplift scores from the uplift model.
    ks : list
        List of fractions, each in (0, 1], representing % of customers targeted.
    margin : float, optional
        Profit per incremental conversion (default: 10.0 USD).
    cost_email : float, optional
        Cost per email sent (default: 0.10 USD).

    Returns
    -------
    pd.DataFrame
        Table containing metrics for each k:
        - k
        - uplift_at_k
        - n_mailed
        - incremental_conv
        - revenue_gain
        - email_cost
        - net_profit
    """
    n = len(y_true)
    results = []

    for k in ks:
        uplift_k = uplift_at_k(y_true, treatment, uplift_scores, k)
        n_targeted = int(np.floor(n * k))

        incremental_conv = uplift_k * n_targeted
        revenue_gain = incremental_conv * margin
        email_cost = n_targeted * cost_email
        net_profit = revenue_gain - email_cost

        results.append(
            {
                "k": k,
                "uplift_at_k": uplift_k,
                "n_mailed": n_targeted,
                "incremental_conv": incremental_conv,
                "revenue_gain": revenue_gain,
                "email_cost": email_cost,
                "net_profit": net_profit,
            }
        )

    return pd.DataFrame(results)
