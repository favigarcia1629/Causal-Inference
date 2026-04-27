# Causal Inference: Did the 2014 Minimum Wage Increases Raise Unemployment?

**A Difference-in-Differences study using real FRED data — 9 treatment states vs 15 control states (2012–2016)**

> The naive comparison said yes. The DiD said no. Methodology changed the conclusion entirely.

🔗 **[Live Interactive Dashboard](#)** *(link added after Streamlit deployment)*

---

## The Question

"Minimum wage increases cause unemployment" is one of the most repeated claims in economics. This project tests it with real government data using Nobel Prize-winning methodology — Difference-in-Differences.

---

## Natural Experiment

**Treatment:** 9 states that legislatively raised their minimum wage in January 2014
**Control:** 15 states that remained at the federal floor of $7.25 through 2015
**Outcome:** Monthly state unemployment rate (FRED / BLS)
**Window:** January 2012 – December 2016 (2 years pre, 2+ years post)

| Group | States |
|---|---|
| Treatment | CT, NJ, NY, CA, MN, MD, MI, HI, RI |
| Control | TX, GA, AL, SC, NC, TN, MS, LA, IN, UT, ID, KS, VA, NH, WY |

---

## Key Results

| Method | Estimate | Interpretation |
|---|---|---|
| **Naive comparison** | +0.244pp | Suggests MW *raised* unemployment |
| **DiD estimate** | **−0.735pp** | MW states fell further than control |
| **P-value** | **< 0.001** | Statistically significant |

**The naive comparison had the direction completely wrong.** Treatment states already had higher pre-policy unemployment — once you control for pre-existing trends, the minimum wage states outperformed the control group.

---

## Why DiD Works

```
DiD = (Treatment After − Treatment Before) − (Control After − Control Before)
    = (5.34 − 7.91) − (5.10 − 6.93)
    = −2.57 − (−1.83)
    = −0.735 percentage points
```

The control group serves as a **counterfactual** — showing what would have happened to treatment states without the policy. The parallel pre-trends (validated in the data) make this comparison credible.

---

## Dashboard Features

- **Parallel Trends** — validate the DiD assumption visually
- **Naive vs DiD** — see exactly how the naive approach misleads
- **Regression results** — OLS with state fixed effects and HC3 robust standard errors
- **State heterogeneity** — DiD estimate broken down by each treatment state
- **Raw data explorer** — all 24 state unemployment series from FRED

---

## Project Structure

```
causal_inference/
├── data/
│   └── fetch.py           # FRED API data fetching + panel construction
├── analysis/
│   └── did.py             # Manual DiD, regression, parallel trends, state effects
├── visualizations/
│   └── charts.py          # Static chart exports (matplotlib)
├── exports/               # LinkedIn-ready PNGs + PDF report (gitignored)
├── app.py                 # Streamlit dashboard
├── main.py                # Headless run: fetch + analyze + export charts
└── generate_pdf.py        # Full PDF report generator
```

---

## Run Locally

```bash
git clone https://github.com/favigarcia1629/causal-inference-minwage.git
cd causal-inference-minwage
pip install -r requirements.txt

# Add your FRED API key (free at fred.stlouisfed.org)
echo "FRED_API_KEY=your_key_here" > .env

# Fetch data, run analysis, export charts
python main.py

# Launch dashboard
streamlit run app.py

# Generate PDF report
python generate_pdf.py
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.14 | Core language |
| FRED API (fredapi) | Real monthly state unemployment data (BLS) |
| pandas / numpy | Panel data construction |
| statsmodels | OLS with fixed effects + HC3 robust SEs |
| plotly | Interactive dashboard charts |
| matplotlib / seaborn | Static export charts |
| Streamlit | Web dashboard + deployment |
| reportlab | PDF report generation |

---

## Methodology Notes

- **Parallel trends assumption** tested and validated visually (pre-2014 data)
- **State fixed effects** absorb time-invariant differences between states
- **HC3 robust standard errors** correct for heteroskedasticity across states
- **No imputation** — only states with complete FRED series included

---

*Data: FRED API (Bureau of Labor Statistics). This replicates the methodology of Card & Krueger (1994) and the framework formalized by Angrist & Imbens (Nobel Prize 2021). Not policy advice — built for research and education.*
