
"""Attempts to use scikit‑uplift (or fetch_hillstrom if
available); otherwise downloads a CSV from a public GitHub mirror"""

# Works with fetch_hillstorm
import pandas as pd

def load_hillstrom() -> pd.DataFrame:
    """Load Hillstrom dataset (from scikit-uplift or fallback CSV)."""
    try:
        from sklift.datasets import fetch_hillstrom
        data = fetch_hillstrom(target_col="all", return_X_y_t=False)
        df = pd.concat([data.data, data.target, data.treatment], axis=1).copy()
        df.columns = [c.lower() for c in df.columns]
    except Exception:
        url = (
            "https://www.minethatdata.com/"
            "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
        )
        df = pd.read_csv(url)
        df.columns = [c.lower() for c in df.columns]
    
    # Check lacking of schema
    required = {"segment", "conversion", "visit", "spend"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Lacking: {missing}. Use sklift.fetch_hillstrom(target_col='all').")
    
    df["conversion"] = df["conversion"].astype(int)
    df["visit"] = df["visit"].astype(int)
    return df


def prepare_treatment(df: pd.DataFrame, treat: str, control: str) -> pd.DataFrame:
    """Prepare dataset for 2-arm uplift training."""
    sub = df[df["segment"].isin([treat, control])].copy()
    sub["treatment"] = (sub["segment"] == treat).astype(int)
    return sub.reset_index(drop=True)
