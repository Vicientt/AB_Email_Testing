"""
Run uplift modeling + Qini curve + ROI simulation.
This script performs:
- Prepare (treatment vs control)
- Train T-learner uplift model
- Draw Qini curve
- Compute ROI at different k%
"""

from src.data.data_loader import load_hillstrom, prepare_treatment
from src.models.uplift_model import train_uplift_tlearner, plot_qini_vs_k
from src.models.roi import simulate_roi


def run_uplift(df, treat, control):
    print(f"\n==============================")
    print(f"UPLIFT MODELING: {treat} vs {control}")
    print(f"==============================")

    # Prepare data
    sub = prepare_treatment(df, treat, control)

    # Choose feature columns
    exclude = {"conversion", "treatment", "segment", "visit", "spend"}
    features = [c for c in sub.columns if c not in exclude]

    # Train model
    res = train_uplift_tlearner(sub, features)

    # Save Qini curve
    save_path = f"figures/qini_{treat.replace(' ','_').lower()}_vs_{control.replace(' ','_').lower()}.png"
    plot_qini_vs_k(
        res["y_true"], 
        res["uplift_pred"],
        res["treatment"],
        title=f"Qini Curve: {treat} vs {control}",
        save_path=save_path
    )
    print(f"Saved Qini curve → {save_path}")

    # ROI simulation
    ks = [0.05, 0.10, 0.20, 0.30, 1.00]
    roi_df = simulate_roi(
        res["y_true"].values,
        res["treatment"].values,
        res["uplift_pred"].values,
        ks,
        margin=15.0,
        cost_email=0.10,
    )
    print("\nROI Simulation:")
    print(roi_df.to_string(index=False))


if __name__ == "__main__":
    df = load_hillstrom()

    # Run for Mens and Womens
    run_uplift(df, "Mens E-Mail", "No E-Mail")
    run_uplift(df, "Womens E-Mail", "No E-Mail")

    print("\nDONE! Uplift modeling completed.")
