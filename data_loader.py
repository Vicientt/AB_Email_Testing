"""
Data loading utilities for the Hillstrom e‑mail experiment.

This module provides functions to load the Hillstrom dataset into a pandas
DataFrame.  It attempts to use ``scikit‑uplift``'s ``fetch_hillstrom`` if
available; otherwise it downloads a CSV from a public GitHub mirror.  The
returned DataFrame includes the original features, target columns and
treatment labels.
"""

import pandas as pd

def load_hillstrom() -> pd.DataFrame:
    try:
        from sklift.datasets import fetch_hillstrom
        # Lấy đủ 3 target: visit, conversion, spend
        data = fetch_hillstrom(target_col="all", return_X_y_t=False)
        df = pd.concat([data.data, data.target, data.treatment], axis=1).copy()
        df.columns = [c.lower() for c in df.columns]
    except Exception:
        url = "https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
        df = pd.read_csv(url)
        # Chuẩn hoá tên cột
        df.columns = [c.lower() for c in df.columns]
        # Một số mirror viết hoa vài cột — đổi về đúng tên
        ren = {"segment": "segment", "history": "history"}
        df = df.rename(columns=ren)

    # Kiểm tra schema tối thiểu
    required = {"segment", "conversion", "visit", "spend"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}. Hãy dùng sklift.fetch_hillstrom(target_col='all').")

    # Ép kiểu nhãn (đề phòng kiểu object)
    df["conversion"] = df["conversion"].astype(int)
    df["visit"] = df["visit"].astype(int)
    return df

def prepare_treatment(df: pd.DataFrame, treat_label: str, control_label: str) -> pd.DataFrame:
    sub = df[df["segment"].isin([treat_label, control_label])].copy()
    sub["treatment"] = (sub["segment"] == treat_label).astype(int)
    return sub.reset_index(drop=True)
