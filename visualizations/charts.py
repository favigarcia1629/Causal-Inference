import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

EXPORT_DIR = Path(__file__).parent.parent / "exports"
EXPORT_DIR.mkdir(exist_ok=True)

TREAT_COLOR = "#E74C3C"
CTRL_COLOR  = "#3498DB"
CF_COLOR    = "#E74C3C"

STYLE = {
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#0F1117",
    "axes.edgecolor":   "#333333",
    "axes.labelcolor":  "#CCCCCC",
    "xtick.color":      "#CCCCCC",
    "ytick.color":      "#CCCCCC",
    "text.color":       "#FFFFFF",
    "grid.color":       "#222222",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
}


def apply_style():
    plt.rcParams.update(STYLE)


def chart1_parallel_trends(trends: pd.DataFrame, save: bool = True) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(13, 6))

    for group, color in [("Treatment (raised MW)", TREAT_COLOR),
                          ("Control (held at $7.25)", CTRL_COLOR)]:
        subset = trends[trends["group"] == group]
        ax.plot(subset["date"], subset["unemp_rate"],
                color=color, linewidth=2.2, label=group)

    ax.axvline(pd.Timestamp("2014-01-01"), color="#FFD700", linewidth=2,
               linestyle="--", label="Policy: Jan 2014")
    ax.fill_betweenx([0, 12], pd.Timestamp("2012-01-01"), pd.Timestamp("2014-01-01"),
                     alpha=0.05, color=CTRL_COLOR)
    ax.fill_betweenx([0, 12], pd.Timestamp("2014-01-01"), pd.Timestamp("2016-12-01"),
                     alpha=0.05, color=TREAT_COLOR)
    ax.text(pd.Timestamp("2012-08-01"), 1.5, "PRE-POLICY", color="#AAAAAA",
            fontsize=9, alpha=0.7)
    ax.text(pd.Timestamp("2014-08-01"), 1.5, "POST-POLICY", color="#AAAAAA",
            fontsize=9, alpha=0.7)

    ax.set_title("Parallel Trends Test — Did Groups Move Together Before 2014?",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel("Unemployment Rate (%)")
    ax.set_xlabel("")
    ax.legend(fontsize=10, framealpha=0.15)
    ax.grid(True)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    if save:
        fig.savefig(EXPORT_DIR / "01_parallel_trends.png", dpi=150, bbox_inches="tight")
    return fig


def chart2_naive_vs_did(did_vals: dict, save: bool = True) -> plt.Figure:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Naive Comparison vs. Difference-in-Differences",
                 fontsize=15, fontweight="bold")

    # Left: Naive bar chart
    ax = axes[0]
    bars = ax.bar(
        ["Control\n(No MW Increase)", "Treatment\n(MW Increase)"],
        [did_vals["ctrl_after"], did_vals["treat_after"]],
        color=[CTRL_COLOR, TREAT_COLOR], alpha=0.85, width=0.5,
        edgecolor="#333"
    )
    ax.set_title("Naive: Just Compare Post-Policy Unemployment",
                 fontweight="bold", color=TREAT_COLOR, fontsize=11)
    ax.set_ylabel("Unemployment Rate (%)")
    ax.set_ylim(0, did_vals["ctrl_after"] * 1.5)
    ax.grid(True, axis="y", alpha=0.4)
    for bar, val in zip(bars, [did_vals["ctrl_after"], did_vals["treat_after"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}%", ha="center", fontsize=12, fontweight="bold")
    naive = did_vals["treat_after"] - did_vals["ctrl_after"]
    ax.annotate(f"Naive gap: {naive:+.3f}pp\n(Misleading — ignores\npre-existing differences)",
                xy=(0.5, 0.12), xycoords="axes fraction", ha="center",
                fontsize=10, color=TREAT_COLOR,
                bbox=dict(boxstyle="round", facecolor="#2D1B1B", alpha=0.8))

    # Right: DiD line chart
    ax = axes[1]
    periods = ["Before 2014", "After 2014"]
    ax.plot(periods, [did_vals["ctrl_before"], did_vals["ctrl_after"]],
            "o-", color=CTRL_COLOR, linewidth=2.5, markersize=10, label="Control")
    ax.plot(periods, [did_vals["treat_before"], did_vals["treat_after"]],
            "o-", color=TREAT_COLOR, linewidth=2.5, markersize=10, label="Treatment")
    ax.plot(periods, [did_vals["treat_before"], did_vals["counterfactual"]],
            "o--", color=CF_COLOR, linewidth=2, markersize=8, alpha=0.5,
            label="Counterfactual (no policy)")

    # DiD bracket
    x1 = 1
    ax.annotate("", xy=(x1, did_vals["treat_after"]),
                xytext=(x1, did_vals["counterfactual"]),
                arrowprops=dict(arrowstyle="<->", color="#2ECC71", lw=2.5))
    mid = (did_vals["treat_after"] + did_vals["counterfactual"]) / 2
    ax.text(1.05, mid, f"DiD = {did_vals['did_estimate']:+.3f}pp",
            fontsize=11, color="#2ECC71", fontweight="bold")

    ax.set_title("DiD: Isolates the True Causal Effect",
                 fontweight="bold", color="#2ECC71", fontsize=11)
    ax.set_ylabel("Unemployment Rate (%)")
    ax.legend(fontsize=9, framealpha=0.15)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    if save:
        fig.savefig(EXPORT_DIR / "02_naive_vs_did.png", dpi=150, bbox_inches="tight")
    return fig


def chart3_regression_coefs(model, save: bool = True) -> plt.Figure:
    apply_style()
    key_params = {
        "post\n(Time trend)":          ("post",         CTRL_COLOR),
        "treated:post\n(DiD effect)":  ("treated:post", "#2ECC71"),
    }
    coefs  = [model.params[k]   for _, (k, _) in key_params.items()]
    errors = [model.bse[k]*1.96 for _, (k, _) in key_params.items()]
    colors = [c for _, (_, c) in key_params.items()]
    labels = list(key_params.keys())
    pvals  = [model.pvalues[k]  for _, (k, _) in key_params.items()]

    fig, ax = plt.subplots(figsize=(10, 5))
    y = range(len(labels))
    ax.barh(list(y), coefs, xerr=errors, color=colors, alpha=0.85,
            capsize=6, height=0.5, edgecolor="#333")
    ax.axvline(0, color="#666666", linewidth=1.5, linestyle="--")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_title("DiD Regression Coefficients (95% CI, HC3 robust SEs)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Effect on Unemployment Rate (percentage points)")
    ax.grid(True, axis="x", alpha=0.4)

    for i, (coef, err, pval) in enumerate(zip(coefs, errors, pvals)):
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else "ns"
        ax.text(coef + err + 0.02, i, f"{coef:+.3f} {sig}",
                va="center", fontsize=9, color="#EEEEEE")

    ax.annotate("*** p<0.01  ** p<0.05  * p<0.1  ns = not significant",
                xy=(0.02, 0.04), xycoords="axes fraction", fontsize=8, color="#888888")
    fig.tight_layout()
    if save:
        fig.savefig(EXPORT_DIR / "03_regression_coefs.png", dpi=150, bbox_inches="tight")
    return fig


def chart4_state_heterogeneity(state_did: pd.DataFrame, save: bool = True) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#2ECC71" if v <= 0 else "#E74C3C" for v in state_did["DiD Estimate"]]
    bars = ax.barh(state_did["State"], state_did["DiD Estimate"],
                   color=colors, alpha=0.85, edgecolor="#333", height=0.6)
    ax.axvline(0, color="#666", linewidth=1.5, linestyle="--")
    ax.set_title("DiD Estimate by State — Heterogeneous Effects of Minimum Wage",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("DiD Estimate (pp change in unemployment vs control group)")
    ax.grid(True, axis="x", alpha=0.4)
    for bar, val in zip(bars, state_did["DiD Estimate"]):
        ax.text(val + (0.02 if val >= 0 else -0.02), bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right",
                fontsize=9, fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(EXPORT_DIR / "04_state_heterogeneity.png", dpi=150, bbox_inches="tight")
    return fig


def export_all(trends, did_vals, model, state_did):
    print("Generating charts...")
    chart1_parallel_trends(trends)
    chart2_naive_vs_did(did_vals)
    chart3_regression_coefs(model)
    chart4_state_heterogeneity(state_did)
    print(f"Charts saved to: {EXPORT_DIR}")
