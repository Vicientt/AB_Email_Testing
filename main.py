# .venv\Scripts\activate
import numpy as np
import pandas as pd
from data_loader import load_hillstrom, prepare_treatment
from eda import check_randomization
from ab_test import ab_test_proportion, ab_test_spend
from uplift_model import train_uplift_tlearner, qini_curve, qini_auc
from roi import simulate_roi
from uplift_model import plot_qini_vs_k

def run_ab_tests(df: pd.DataFrame):
    # Conversion z-tests
    def counts(seg_a, seg_b):
        sub = df[df["segment"].isin([seg_a, seg_b])]
        tab = sub.groupby("segment")["conversion"].agg(["sum", "count"])
        return int(tab.loc[seg_a, "sum"]), int(tab.loc[seg_a, "count"]), int(tab.loc[seg_b, "sum"]), int(tab.loc[seg_b, "count"])

    pairs = [("Mens E-Mail", "No E-Mail"),
             ("Womens E-Mail", "No E-Mail"),
             ("Mens E-Mail", "Womens E-Mail")]

    for a, b in pairs:
        if {a, b}.issubset(set(df["segment"].unique())):
            x_s, x_n, y_s, y_n = counts(a, b)
            print(f"\nConversion: {a} vs {b}")
            print(ab_test_proportion(x_s, x_n, y_s, y_n))  # z-test results

    # Spend test (optional)
    if "spend" in df.columns:
        for a, b in [("Mens E-Mail", "No E-Mail"), ("Womens E-Mail", "No E-Mail")]:
            xa = df.loc[df["segment"] == a, "spend"].to_numpy()
            xb = df.loc[df["segment"] == b, "spend"].to_numpy()
            res = ab_test_spend(xa, xb, n_boot=5000, seed=42)  # Welch + bootstrap CI
            print(f"\nSpend: {a} vs {b} (Welch + bootstrap CI)")
            print(res)

def run_uplift_and_roi(df: pd.DataFrame, treat_label: str, control_label: str):
    # Chuẩn bị 2-nhánh (treatment vs control)
    sub = prepare_treatment(df, treat_label, control_label)
    # Chọn feature tự động (loại bỏ target/cols không dùng)
    exclude = {"conversion", "treatment", "segment", "visit", "spend"}
    features = [c for c in sub.columns if c not in exclude]

    # Train T-learner RF và dự đoán uplift
    res = train_uplift_tlearner(sub, features)  # -> DataFrame: y_true, treatment, uplift_pred

    # Qini curve & AUC
    phi, qini = qini_curve(res["y_true"].values, res["uplift_pred"].values, res["treatment"].values)
    qini_auc_val = qini_auc(phi, qini, res["treatment"].values, res["y_true"].values)
    print(f"\nQini AUC ({treat_label} vs {control_label}):", qini_auc_val)
    
    # Draw Qini Curve
    plot_path = f"figures/qini_{treat_label.replace(' ','_').lower()}_vs_{control_label.replace(' ','_').lower()}.png"
    plot_qini_vs_k(
    res["y_true"].values,
    res["uplift_pred"].values,
    res["treatment"].values,
    title=f"Qini Curve: {treat_label} vs {control_label} (AUC={qini_auc_val:.3f})",
    save_path=plot_path,
    )
    print("Saved Qini curve to:", plot_path)

    # ROI mô phỏng cho các k
    ks = [0.05, 0.10, 0.20, 0.30, 1.00]
    roi_df = simulate_roi(res["y_true"].values, res["treatment"].values, res["uplift_pred"].values,
                          ks, margin=15.0, cost_email=0.10)
    print("\nROI (send top-k%):")
    print(roi_df.to_string(index=False))

def main():
    df = load_hillstrom()
    # EDA: kiểm tra cân bằng ngẫu nhiên (nếu các cột tồn tại)
    base_cols = [c for c in ["history", "recency"] if c in df.columns]
    if base_cols:
        print("Randomization check:")
        print(check_randomization(df, base_cols).head())

    # A/B tests
    run_ab_tests(df)

    # Uplift + ROI cho Mens vs Control (có thể lặp lại cho Womens nếu muốn)
    for treat in ["Mens E-Mail", "Womens E-Mail"]:
        run_uplift_and_roi(df, treat, "No E-Mail")
    
    print("\nColumns:", list(df.columns))

if __name__ == "__main__":
    main()
