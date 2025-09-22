"""
A/B testing utilities for the Hillstrom experiment.

This module provides functions to perform z‑tests for conversion rates and
Welch’s t‑tests with bootstrap confidence intervals for spend data.  The
tests return test statistics, p‑values and estimates of absolute and
relative lifts.
"""

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


def ab_test_proportion(x_success: int, x_total: int, y_success: int, y_total: int):
    """Perform a two‑sample z‑test for proportions.

    Parameters
    ----------
    x_success, y_success : int
        Number of successes (conversions) in each group.
    x_total, y_total : int
        Total number of observations in each group.

    Returns
    -------
    dict
        z‑statistic, p‑value, absolute lift, relative lift, and group rates.
    """
    count = np.array([x_success, y_success])
    nobs = np.array([x_total, y_total])
    # Use statsmodels to perform a two-sample z‑test for proportions
    stat, pval = proportions_ztest(count, nobs)
    # Estimated conversion rates for each group
    rates = count / nobs
    rate_x, rate_y = rates[0], rates[1]
    abs_lift = rate_x - rate_y
    rel_lift = abs_lift / rate_y if rate_y > 0 else np.nan
    return {
        "z_stat": stat,
        "p_value": pval,
        "abs_lift": abs_lift,
        "rel_lift": rel_lift,
        "rate_x": rate_x,
        "rate_y": rate_y,
    }


def ab_test_spend(x_vals: np.ndarray, y_vals: np.ndarray, n_boot: int = 5000, seed: int = 42):
    """Perform Welch’s t‑test and a bootstrap confidence interval on spend.

    Parameters
    ----------
    x_vals, y_vals : array-like
        Spend values for treatment and control groups.
    n_boot : int, optional
        Number of bootstrap resamples for the percentile confidence interval.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        t‑statistic, p‑value, bootstrap 95 % confidence interval and mean difference.
    """
    # Tackle null values
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    x_vals = x_vals[~np.isnan(x_vals)]
    y_vals = y_vals[~np.isnan(y_vals)]

    t_stat, p_val = stats.ttest_ind(x_vals, y_vals, equal_var=False)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sample_x = rng.choice(x_vals, size=len(x_vals), replace=True)
        sample_y = rng.choice(y_vals, size=len(y_vals), replace=True)
        diffs.append(sample_x.mean() - sample_y.mean())
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return {
        "t_stat": t_stat,
        "p_value": p_val,
        "ci_lower": lower,
        "ci_upper": upper,
        "mean_diff": x_vals.mean() - y_vals.mean(),
    }