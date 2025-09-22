"""
Exploratory data analysis (EDA) utilities for the Hillstrom dataset.

This module contains helper functions to compute descriptive statistics
for baseline variables by segment.  Such statistics are useful for
checking that the random assignment of e‑mails produced balanced groups.
"""

import pandas as pd


def check_randomization(df: pd.DataFrame, baseline_cols: list) -> pd.DataFrame:
    """Compute means and standard deviations of baseline variables by segment.

    Parameters
    ----------
    df : pd.DataFrame
        The Hillstrom dataset with a ``segment`` column.
    baseline_cols : list
        List of column names to summarize (e.g., ``["history", "recency"]``).

    Returns
    -------
    pd.DataFrame
        Tidy table with segment, variable, mean and standard deviation.
    """
    stats_df = (
        df.groupby("segment")[baseline_cols]
        .agg(["mean", "std"])
        .stack(future_stack=True)
        .reset_index()
        .rename(columns={"level_1": "variable", "mean": "mean", "std": "std"})
    )
    return stats_df