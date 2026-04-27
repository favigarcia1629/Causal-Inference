import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.fetch import fetch_unemployment, build_panel
from analysis.did import compute_group_means, manual_did, run_did_regression, parallel_trends, state_level_did
from visualizations.charts import export_all

if __name__ == "__main__":
    print("Fetching unemployment data from FRED...")
    df    = fetch_unemployment()
    panel = build_panel(df)

    print("\n=== GROUP MEANS ===")
    print(compute_group_means(panel).to_string())

    did_vals = manual_did(panel)
    print("\n=== MANUAL DiD ===")
    print(f"Treatment before:  {did_vals['treat_before']:.3f}%")
    print(f"Treatment after:   {did_vals['treat_after']:.3f}%")
    print(f"Control before:    {did_vals['ctrl_before']:.3f}%")
    print(f"Control after:     {did_vals['ctrl_after']:.3f}%")
    print(f"DiD estimate:      {did_vals['did_estimate']:+.4f}pp")
    print(f"Naive estimate:    {did_vals['naive_gap']:+.4f}pp")

    model = run_did_regression(panel)
    print("\n=== REGRESSION (key coefficients) ===")
    for k in ["post", "treated:post"]:
        print(f"  {k:20s}: {model.params[k]:+.4f}  (p={model.pvalues[k]:.4f})")

    print("\n=== STATE-LEVEL DiD ===")
    print(state_level_did(panel).to_string(index=False))

    trends = parallel_trends(panel)
    state_did_df = state_level_did(panel)
    export_all(trends, did_vals, model, state_did_df)
    print("\nDone. Run: streamlit run app.py")
