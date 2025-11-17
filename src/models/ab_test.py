import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

def ab_test_proportion(x_success, x_total, y_success, y_total):
    count = np.array([x_success, y_success])
    nobs = np.array([x_total, y_total])
    z_stat, p_val = proportions_ztest(count, nobs)

    rate_x, rate_y = count[0]/nobs[0], count[1]/nobs[1]
    abs_lift = rate_x - rate_y
    rel_lift = abs_lift / rate_y if rate_y > 0 else np.nan

    return {
        "z_stat": z_stat,
        "p_value": p_val,
        "rate_x": rate_x,
        "rate_y": rate_y,
        "abs_lift": abs_lift,
        "rel_lift": rel_lift,
    }


def ab_test_spend(x_vals, y_vals, n_boot=5000, seed=42):
    x_vals, y_vals = np.asarray(x_vals), np.asarray(y_vals)
    x_vals, y_vals = x_vals[~np.isnan(x_vals)], y_vals[~np.isnan(y_vals)]

    t_stat, p_val = stats.ttest_ind(x_vals, y_vals, equal_var=False)

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sx = rng.choice(x_vals, size=len(x_vals), replace=True)
        sy = rng.choice(y_vals, size=len(y_vals), replace=True)
        diffs.append(sx.mean() - sy.mean())
    lower, upper = np.percentile(diffs, [2.5, 97.5])

    return {
        "t_stat": t_stat,
        "p_value": p_val,
        "mean_diff": x_vals.mean() - y_vals.mean(),
        "ci_lower": lower,
        "ci_upper": upper,
    }
