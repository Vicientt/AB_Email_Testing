from src.data.data_loader import load_hillstrom, prepare_treatment
from src.features.eda import check_randomization
from src.models.ab_test import ab_test_proportion, ab_test_spend
from src.models.uplift_model import (
    train_uplift_tlearner,
    qini_curve,
    qini_auc,
    plot_qini_vs_k,
)
from src.models.roi import simulate_roi

def run_ab_tests(df):
    def counts(a, b):
        sub = df[df["segment"].isin([a, b])]
        tab = sub.groupby("segment")["conversion"].agg(["sum", "count"])
        return tab.loc[a, "sum"], tab.loc[a, "count"], tab.loc[b, "sum"], tab.loc[b, "count"]

    pairs = [("Mens E-Mail", "No E-Mail"),
             ("Womens E-Mail", "No E-Mail"),
             ("Mens E-Mail", "Womens E-Mail")]

    for a, b in pairs:
        x_s, x_n, y_s, y_n = counts(a, b)
        print(f"\nConversion: {a} vs {b}")
        print(ab_test_proportion(x_s, x_n, y_s, y_n))

    for a, b in [("Mens E-Mail", "No E-Mail"), ("Womens E-Mail", "No E-Mail")]:
        xa = df[df["segment"] == a]["spend"].to_numpy()
        xb = df[df["segment"] == b]["spend"].to_numpy()
        print(f"\nSpend: {a} vs {b}")
        print(ab_test_spend(xa, xb))


def run_uplift(df, treat, control):
    sub = prepare_treatment(df, treat, control)
    exclude = {"conversion", "treatment", "segment", "visit", "spend"}
    features = [c for c in sub.columns if c not in exclude]

    res = train_uplift_tlearner(sub, features)

    phi, qini = qini_curve(res["y_true"], res["uplift_pred"], res["treatment"])
    auc = qini_auc(phi, qini, res["treatment"], res["y_true"])
    print(f"\nQini AUC ({treat} vs {control}):", auc)

    img_path = f"figures/qini_{treat.replace(' ','_').lower()}_vs_{control.replace(' ','_').lower()}.png"
    plot_qini_vs_k(res["y_true"], res["uplift_pred"], res["treatment"], save_path=img_path)
    print("Saved:", img_path)

    ks = [0.05, 0.10, 0.20, 0.30, 1.00]
    roi_df = simulate_roi(
        res["y_true"].values,
        res["treatment"].values,
        res["uplift_pred"].values,
        ks,
        margin=15.0,
        cost_email=0.10,
    )
    print("\nROI:")
    print(roi_df)

def main():
    df = load_hillstrom()

    base_cols = [c for c in ["history", "recency"] if c in df.columns]
    print(check_randomization(df, base_cols).head())

    run_ab_tests(df)

    for treat in ["Mens E-Mail", "Womens E-Mail"]:
        run_uplift(df, treat, "No E-Mail")

if __name__ == "__main__":
    main()
