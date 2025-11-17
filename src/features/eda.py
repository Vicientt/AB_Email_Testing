import pandas as pd

def check_randomization(df: pd.DataFrame, baseline_cols: list) -> pd.DataFrame:
    return (
        df.groupby("segment")[baseline_cols]
        .agg(["mean", "std"])
        .stack(future_stack=True)
        .reset_index()
        .rename(columns={"level_1": "variable"})
    )