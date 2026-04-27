import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from data.fetch import fetch_unemployment, build_panel, TREATMENT_STATES, CONTROL_STATES, MIN_WAGE_INCREASES
from analysis.did import (
    compute_group_means, manual_did, run_did_regression,
    parallel_trends, state_level_did
)

st.set_page_config(
    page_title="Causal Inference: Minimum Wage & Employment",
    page_icon="⚖️",
    layout="wide",
)

TREAT_COLOR = "#E74C3C"
CTRL_COLOR  = "#3498DB"
GREEN       = "#2ECC71"

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#FFFFFF; margin-bottom:4px'>
    ⚖️ Causal Inference: Did Minimum Wage Increases Raise Unemployment?
</h1>
<p style='text-align:center; color:#AAAAAA; font-size:1.05rem'>
    A Difference-in-Differences study using real FRED data — 9 treatment states vs 15 control states (2012–2016)
</p>
<hr style='border:1px solid #333; margin:16px 0'>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching state unemployment data from FRED...")
def load():
    df    = fetch_unemployment()
    panel = build_panel(df)
    trends = parallel_trends(panel)
    did_vals = manual_did(panel)
    model    = run_did_regression(panel)
    state_did_df = state_level_did(panel)
    return df, panel, trends, did_vals, model, state_did_df

df, panel, trends, did_vals, model, state_did_df = load()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ About This Study")
    st.markdown("""
    **Natural Experiment:**
    Several U.S. states raised their minimum wage in **January 2014**.
    States that kept the federal floor of **$7.25** serve as the control group.

    **Method:** Difference-in-Differences (DiD)
    Isolates the *causal* effect of the policy by removing trends
    that would have happened anyway.

    **Outcome variable:** State unemployment rate (monthly, FRED)
    """)
    st.markdown("---")
    st.markdown("**Treatment States** (raised MW in 2014)")
    for abbr, name in TREATMENT_STATES.items():
        w = MIN_WAGE_INCREASES[abbr]
        st.markdown(f"- {name}: ${w['before']} → ${w['after']}")
    st.markdown("---")
    st.markdown("**Control States** (held at $7.25)")
    for abbr, name in CONTROL_STATES.items():
        st.markdown(f"- {name}")
    st.markdown("---")
    st.caption("Data: FRED API (Bureau of Labor Statistics) · 2012–2016")

# ── KPI Row ───────────────────────────────────────────────────────────────────
did_est  = did_vals["did_estimate"]
did_pval = model.pvalues["treated:post"]
did_se   = model.bse["treated:post"]
naive    = did_vals["naive_gap"]

st.subheader("Key Results")
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div style='background:#1A2634;border-left:4px solid {TREAT_COLOR};padding:14px;border-radius:8px'>
    <div style='color:{TREAT_COLOR};font-weight:700'>DiD Estimate</div>
    <div style='font-size:2rem;font-weight:800;color:#FFF'>{did_est:+.3f}pp</div>
    <div style='color:#AAA;font-size:0.8rem'>causal effect on unemployment</div>
    </div>""", unsafe_allow_html=True)
with k2:
    sig_color = GREEN if did_pval < 0.05 else "#FFD700"
    sig_label = "Statistically Significant" if did_pval < 0.05 else "Not Significant"
    st.markdown(f"""
    <div style='background:#1A2634;border-left:4px solid {sig_color};padding:14px;border-radius:8px'>
    <div style='color:{sig_color};font-weight:700'>P-Value</div>
    <div style='font-size:2rem;font-weight:800;color:#FFF'>{did_pval:.4f}</div>
    <div style='color:#AAA;font-size:0.8rem'>{sig_label}</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div style='background:#1A2634;border-left:4px solid {CTRL_COLOR};padding:14px;border-radius:8px'>
    <div style='color:{CTRL_COLOR};font-weight:700'>Naive Estimate</div>
    <div style='font-size:2rem;font-weight:800;color:#FFF'>{naive:+.3f}pp</div>
    <div style='color:#AAA;font-size:0.8rem'>biased — ignores pre-trends</div>
    </div>""", unsafe_allow_html=True)
with k4:
    bias = abs(naive - did_est)
    st.markdown(f"""
    <div style='background:#1A2634;border-left:4px solid #FFD700;padding:14px;border-radius:8px'>
    <div style='color:#FFD700;font-weight:700'>Naive Bias</div>
    <div style='font-size:2rem;font-weight:800;color:#FFF'>{bias:.3f}pp</div>
    <div style='color:#AAA;font-size:0.8rem'>overstatement without DiD</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Parallel Trends", "⚖️ Naive vs DiD", "📊 Regression", "🗺️ State Effects", "📋 Data"
])

# ── Tab 1: Parallel Trends ────────────────────────────────────────────────────
with tab1:
    st.markdown("### The Parallel Trends Assumption")
    st.markdown(
        "For DiD to be valid, treatment and control groups must have moved "
        "**in parallel before the policy**. If they were already diverging, "
        "we can't attribute post-2014 differences to the minimum wage."
    )
    fig = go.Figure()
    for group, color in [("Treatment (raised MW)", TREAT_COLOR),
                          ("Control (held at $7.25)", CTRL_COLOR)]:
        subset = trends[trends["group"] == group]
        fig.add_trace(go.Scatter(
            x=subset["date"], y=subset["unemp_rate"],
            name=group, line=dict(color=color, width=2.5),
            hovertemplate="%{y:.2f}%<extra>" + group + "</extra>",
        ))
    fig.add_shape(type="line", x0="2014-01-01", x1="2014-01-01",
                  y0=0, y1=1, yref="paper",
                  line=dict(color="#FFD700", width=2, dash="dash"))
    fig.add_annotation(x="2014-01-01", y=0.98, yref="paper",
                       text="Min Wage Increase (Jan 2014)",
                       font=dict(color="#FFD700", size=11),
                       showarrow=False, xanchor="left", bgcolor="rgba(0,0,0,0.3)")
    fig.add_vrect(x0="2012-01-01", x1="2014-01-01",
                  fillcolor="rgba(33,150,243,0.05)", line_width=0)
    fig.add_annotation(x="2012-06-01", y=0.98, yref="paper",
                       text="Pre-Policy", font=dict(color="#AAAAAA", size=10),
                       showarrow=False)
    fig.add_vrect(x0="2014-01-01", x1="2016-12-01",
                  fillcolor="rgba(231,76,60,0.05)", line_width=0)
    fig.add_annotation(x="2015-01-01", y=0.98, yref="paper",
                       text="Post-Policy", font=dict(color="#AAAAAA", size=10),
                       showarrow=False)
    fig.update_layout(
        template="plotly_dark", height=460,
        yaxis_title="Average Unemployment Rate (%)", xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "**Key observation:** Both groups trended downward together from 2012–2013 "
        "(parallel pre-trends ✅). After 2014, the treatment group's unemployment "
        "declined *differently* — the DiD isolates whether that difference is the policy."
    )

# ── Tab 2: Naive vs DiD ───────────────────────────────────────────────────────
with tab2:
    st.markdown("### Why the Naive Comparison is Wrong")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**❌ Naive: Just compare post-2014 unemployment**")
        fig_naive = go.Figure(go.Bar(
            x=["Control (No MW Increase)", "Treatment (MW Increase)"],
            y=[did_vals["ctrl_after"], did_vals["treat_after"]],
            marker_color=[CTRL_COLOR, TREAT_COLOR],
            text=[f"{did_vals['ctrl_after']:.2f}%", f"{did_vals['treat_after']:.2f}%"],
            textposition="outside",
        ))
        fig_naive.update_layout(
            template="plotly_dark", height=380,
            yaxis_title="Unemployment Rate (%)",
            yaxis_range=[0, max(did_vals["ctrl_after"], did_vals["treat_after"]) * 1.4],
            showlegend=False,
        )
        st.plotly_chart(fig_naive, use_container_width=True)
        st.error(f"Naive gap: **{naive:+.3f}pp** — this is biased because treatment states may have started with higher unemployment.")

    with col2:
        st.markdown("**✅ DiD: Account for pre-existing trends**")
        fig_did = go.Figure()
        periods = ["Before 2014", "After 2014"]
        fig_did.add_trace(go.Scatter(
            x=periods,
            y=[did_vals["ctrl_before"], did_vals["ctrl_after"]],
            name="Control", mode="lines+markers",
            line=dict(color=CTRL_COLOR, width=2.5),
            marker=dict(size=10),
        ))
        fig_did.add_trace(go.Scatter(
            x=periods,
            y=[did_vals["treat_before"], did_vals["treat_after"]],
            name="Treatment", mode="lines+markers",
            line=dict(color=TREAT_COLOR, width=2.5),
            marker=dict(size=10),
        ))
        fig_did.add_trace(go.Scatter(
            x=periods,
            y=[did_vals["treat_before"], did_vals["counterfactual"]],
            name="Counterfactual", mode="lines+markers",
            line=dict(color=TREAT_COLOR, width=2, dash="dash"),
            marker=dict(size=8), opacity=0.6,
        ))
        fig_did.update_layout(
            template="plotly_dark", height=380,
            yaxis_title="Unemployment Rate (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_did, use_container_width=True)
        st.success(f"DiD estimate: **{did_est:+.3f}pp** (p={did_pval:.4f}) — causal effect after removing pre-existing trends.")

# ── Tab 3: Regression ─────────────────────────────────────────────────────────
with tab3:
    st.markdown("### DiD Regression Results")
    st.markdown(
        "Model: `unemployment ~ treated + post + treated×post + state fixed effects`  \n"
        "The `treated×post` coefficient is **the DiD estimator** — the causal effect of the policy."
    )

    key_rows = ["post", "treated:post"]
    reg_table = pd.DataFrame({
        "Coefficient": model.params[key_rows].round(4),
        "Std Error":   model.bse[key_rows].round(4),
        "t-stat":      model.tvalues[key_rows].round(3),
        "P-value":     model.pvalues[key_rows].round(4),
        "95% CI Lower": model.conf_int().loc[key_rows, 0].round(4),
        "95% CI Upper": model.conf_int().loc[key_rows, 1].round(4),
    })
    reg_table.index = ["Post (time trend)", "Treated × Post (DiD causal effect)"]
    st.dataframe(reg_table, use_container_width=True)

    # Coefficient plot
    labels = ["Post\n(Time)", "Treated×Post\n(DiD Effect)"]
    coefs  = [model.params[k] for k in key_rows]
    errors = [model.bse[k] * 1.96 for k in key_rows]
    colors = [CTRL_COLOR, GREEN]

    fig = go.Figure()
    for i, (label, coef, err, color) in enumerate(zip(labels, coefs, errors, colors)):
        fig.add_trace(go.Scatter(
            x=[coef], y=[label], mode="markers",
            error_x=dict(type="data", array=[err], visible=True, color=color),
            marker=dict(color=color, size=12),
            name=label, showlegend=False,
        ))
    fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=1, yref="paper",
                  line=dict(color="#555555", width=1.5, dash="dash"))
    fig.update_layout(
        template="plotly_dark", height=300,
        xaxis_title="Effect on Unemployment Rate (percentage points)",
        title="Regression Coefficients with 95% Confidence Intervals",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Model R²:** {model.rsquared:.3f} | **Observations:** {int(model.nobs)} | **Standard errors:** HC3 (heteroskedasticity-robust)")

# ── Tab 4: State Effects ──────────────────────────────────────────────────────
with tab4:
    st.markdown("### Heterogeneous Effects by State")
    st.markdown(
        "The average DiD hides variation. Some states saw unemployment "
        "fall *more* than control states after raising minimum wage — others fell less. "
        "This reflects differences in local labor markets, industry mix, and the size of the wage increase."
    )
    fig = px.bar(
        state_did_df, x="DiD Estimate", y="State",
        orientation="h",
        color="DiD Estimate",
        color_continuous_scale=["#2ECC71", "#FFFFFF", "#E74C3C"],
        color_continuous_midpoint=0,
        text="DiD Estimate",
        title="DiD Estimate per State (negative = unemployment fell more than control)",
    )
    fig.update_traces(texttemplate="%{text:+.3f}pp", textposition="outside")
    fig.update_layout(template="plotly_dark", height=420, showlegend=False,
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(state_did_df, use_container_width=True)

# ── Tab 5: Raw Data ───────────────────────────────────────────────────────────
with tab5:
    st.markdown("### Raw Unemployment Data (FRED)")
    fig = go.Figure()
    for col in df.columns:
        color = TREAT_COLOR if col in TREATMENT_STATES else CTRL_COLOR
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=col,
            line=dict(color=color, width=1.2),
            opacity=0.7,
            hovertemplate=f"{col}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.add_shape(type="line", x0="2014-01-01", x1="2014-01-01",
                  y0=0, y1=1, yref="paper",
                  line=dict(color="#FFD700", width=2, dash="dash"))
    fig.update_layout(
        template="plotly_dark", height=450,
        yaxis_title="Unemployment Rate (%)", xaxis_title="",
        title="Individual State Unemployment Rates (Red = Treatment, Blue = Control)",
        hovermode="x unified",
        legend=dict(orientation="v", x=1.01),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df.round(2), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr style='border:1px solid #333;margin-top:32px'>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center;color:#666;font-size:0.8rem'>
Data: FRED API (Bureau of Labor Statistics) · Natural experiment: 2014 state minimum wage increases ·
Not policy advice — built for research & education
</p>""", unsafe_allow_html=True)
