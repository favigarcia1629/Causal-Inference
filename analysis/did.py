"""
Difference-in-Differences estimation.

Model: unemp_rate ~ treated + post + treated:post + state (fixed effects)
The treated:post coefficient is the DiD estimator — the causal effect
of the minimum wage increase on unemployment.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from data.fetch import TREATMENT_STATES, CONTROL_STATES, POLICY_DATE


def compute_group_means(panel: pd.DataFrame) -> pd.DataFrame:
    return (
        panel.groupby(["group", "period"])["unemp_rate"]
        .mean()
        .round(3)
        .unstack()
        .reindex(columns=["Pre-2014", "Post-2014"])
    )


def manual_did(panel: pd.DataFrame) -> dict:
    means = panel.groupby(["treated", "post"])["unemp_rate"].mean()

    treat_before = means[1, 0]
    treat_after  = means[1, 1]
    ctrl_before  = means[0, 0]
    ctrl_after   = means[0, 1]

    treat_change = treat_after - treat_before
    ctrl_change  = ctrl_after  - ctrl_before
    did_estimate = treat_change - ctrl_change
    naive_gap    = treat_after - ctrl_after
    counterfactual = treat_before + ctrl_change

    return {
        "treat_before":    treat_before,
        "treat_after":     treat_after,
        "ctrl_before":     ctrl_before,
        "ctrl_after":      ctrl_after,
        "treat_change":    treat_change,
        "ctrl_change":     ctrl_change,
        "did_estimate":    did_estimate,
        "naive_gap":       naive_gap,
        "counterfactual":  counterfactual,
    }


def run_did_regression(panel: pd.DataFrame):
    """
    Two-way Fixed Effects DiD:
      - treated:post  → causal DiD estimate (the key coefficient)
      - C(state)      → state fixed effects (absorbs time-invariant group differences)
      - post          → time fixed effect
    Note: 'treated' alone is collinear with state FE and dropped — that's expected.
    """
    model = smf.ols(
        "unemp_rate ~ post + treated:post + C(state)",
        data=panel
    ).fit(cov_type="HC3")   # Heteroskedasticity-robust standard errors
    return model


def parallel_trends(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly average unemployment by group.
    Pre-2014 trends should be parallel if DiD assumptions hold.
    """
    return (
        panel.groupby(["date", "group"])["unemp_rate"]
        .mean()
        .reset_index()
    )


def state_level_did(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute DiD estimate for each treatment state vs all control states.
    Shows heterogeneity across states.
    """
    ctrl_means = panel[panel["treated"] == 0].groupby("post")["unemp_rate"].mean()
    ctrl_change = ctrl_means[1] - ctrl_means[0]

    rows = []
    for state, name in TREATMENT_STATES.items():
        state_data = panel[panel["state"] == state].groupby("post")["unemp_rate"].mean()
        if len(state_data) < 2:
            continue
        state_change = state_data[1] - state_data[0]
        did = state_change - ctrl_change
        rows.append({
            "State": f"{name} ({state})",
            "Before 2014 (UR%)": round(state_data[0], 2),
            "After 2014 (UR%)":  round(state_data[1], 2),
            "State Change":      round(state_change, 3),
            "Control Change":    round(ctrl_change, 3),
            "DiD Estimate":      round(did, 3),
        })
    return pd.DataFrame(rows).sort_values("DiD Estimate")
