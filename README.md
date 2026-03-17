Causal Inference: Policy Impact Evaluation ⚖️
**Portfolio Project #08 | Econometrics | Python | Difference-in-Differences**

## Results at a Glance

| Method | Estimate | P-Value | Verdict |
|--------|----------|---------|---------|
| Naive Comparison | -0.332% | — | ❌ Biased — ignores pre-existing trend |
| **Difference-in-Differences** | **-0.367%** | **0.017** | ✅ Statistically significant causal effect |
| True Simulated Effect | -0.370% | — | DiD matched within 0.003% |

DiD removed **9.5% bias** from the naive estimate — and matched the true causal effect to within 0.003 percentage points. Methodology validated.

---

## Overview

This project applies **Difference-in-Differences (DiD)** methodology to evaluate the causal effect of a minimum wage increase on employment, replicating the approach of Card and Krueger's 1994 Nobel Prize-winning study.

It demonstrates why naive before/after comparisons systematically overstate or understate policy effects — and how DiD corrects this by using a control group as a **counterfactual**: the answer to the question that economics can never directly observe — *"what would have happened without the policy?"*

> As of 2026, credible causal reasoning is ranked as the #1 in-demand skill for economists by the IMF and World Economic Forum.

---

## The Research Question

**Did the minimum wage increase cause the observed employment decline — or would employment have fallen anyway due to broader economic conditions?**

The naive answer compares treated and control counties after the policy. The correct answer asks: how much did the treated group diverge from the trend the control group reveals?

---

## Dataset

Simulated dataset — 200 counties, 400 observations. No download required.

| Variable | Description | Value |
|----------|-------------|-------|
| `treated` | 1 if county received minimum wage increase | 100 counties |
| `control` | 0 if county did not receive increase | 100 counties |
| Pre-policy employment | Employment rate before intervention | ~8.1–8.2% |
| Post-policy employment | Employment rate after intervention | 7.6–8.0% |
| DiD estimator | True causal effect of policy | **-0.367 percentage points** |

---

## Methodology

**1. Data Simulation**
200 counties simulated — 100 treatment (minimum wage raised), 100 control (no change). Both groups start at similar employment levels (~8.1–8.2%) to satisfy the parallel trends assumption.

**2. Manual DiD Calculation**
```
DiD = (Treatment after − Treatment before) − (Control after − Control before)
DiD = (7.685 − 8.152) − (8.018 − 8.117)
DiD = (−0.466) − (−0.099) = −0.367
```
The control group's decline (−0.099) tells us what would have happened without the policy. Subtracting it isolates the causal effect.

**3. Naive Comparison (for contrast)**
Simple post-period comparison: Treatment (7.685%) − Control (8.018%) = **−0.332%**. This ignores the pre-existing trend and overstates the damage.

**4. OLS Regression Validation**
DiD implemented as a regression with an interaction term:
```
Employment = β₀ + β₁(Treated) + β₂(Post) + β₃(Treated × Post) + ε
```
**β₃ is the DiD estimator.** Result: β₃ = −0.367, **p = 0.017** — statistically significant at the 5% level.

**5. Counterfactual Visualization**
Dashed counterfactual line plotted showing what treatment group employment would have looked like without the policy, based on the control group's trend. The DiD effect is the gap between actual and counterfactual.

---

## Key Findings

- **DiD estimate of −0.367 percentage points (p = 0.017)** — statistically significant causal effect of the minimum wage increase
- **Naive estimate of −0.332% was biased by 9.5%** — the economy was already declining before the policy and that trend must be subtracted out
- **DiD matched the true simulated effect (−0.370%) almost exactly** — validating the methodology against known ground truth
- **Card and Krueger (1994)** used the same approach and found near-zero employment effects from New Jersey's minimum wage increase — a result that contradicted classical theory and contributed to the 2021 Nobel Prize in Economics

---

## How to Run

```bash
pip install pandas numpy statsmodels matplotlib seaborn
# Dataset is generated within the notebook — no external download needed
jupyter notebook causal_inference_policy.ipynb
```

---

## Tools & Libraries

- Python 3.12
- `statsmodels` — OLS regression with interaction terms
- `pandas` / `numpy` — data simulation and manipulation
- `matplotlib` / `seaborn` — counterfactual visualization and coefficient plots

---

## About

Built as part of an Economics portfolio by an economics major exploring machine learning applications in finance and policy. Replicates the methodology of Card and Krueger (1994), which contributed to the 2021 Nobel Prize in Economics.
