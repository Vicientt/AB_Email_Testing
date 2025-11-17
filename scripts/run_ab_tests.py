"""
Run A/B tests for Hillstrom Email Experiment.
This script performs:
- Conversion z-tests
- Spend t-test + bootstrap CI
"""

from src.data.data_loader import load_hillstrom
from src.models.ab_test import ab_test_proportion, ab_test_spend


def run_ab_tests(df):
    """Run all A/B tests."""
    def counts(a, b):
        sub = df[df["segment"].isin([a, b])]
        tab = sub.groupby("segment")["conversion"].agg(["sum", "count"])
        return tab.loc[a, "sum"], tab.loc[a, "count"], tab.loc[b, "sum"], tab.loc[b, "count"]

    pairs = [
        ("Mens E-Mail", "No E-Mail"),
        ("Womens E-Mail", "No E-Mail"),
        ("Mens E-Mail", "Womens E-Mail"),
    ]

    print("\n========================")
    print("A/B TESTING - CONVERSION")
    print("========================")

    for a, b in pairs:
        x_s, x_n, y_s, y_n = counts(a, b)
        print(f"\n{a} vs {b}")
        print(ab_test_proportion(x_s, x_n, y_s, y_n))

    print("\n====================")
    print("A/B TESTING - SPEND")
    print("====================")

    for a, b in [
        ("Mens E-Mail", "No E-Mail"),
        ("Womens E-Mail", "No E-Mail")
    ]:
        x_vals = df[df["segment"] == a]["spend"].to_numpy()
        y_vals = df[df["segment"] == b]["spend"].to_numpy()
        print(f"\n{a} vs {b}")
        print(ab_test_spend(x_vals, y_vals))


if __name__ == "__main__":
    df = load_hillstrom()
    run_ab_tests(df)
    print("\nDONE! A/B testing completed.")
