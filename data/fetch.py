"""
Fetches state-level unemployment rates from FRED.

Natural experiment: 2014 minimum wage increases.
Treatment = states that legislatively raised minimum wage in 2014.
Control   = states that stayed at the federal minimum ($7.25) through 2015.

Window: Jan 2012 – Dec 2016 (2 years pre, 2+ years post).
"""
import os
import pandas as pd
from pathlib import Path
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

START = "2012-01-01"
END   = "2016-12-01"
POLICY_DATE = "2014-01-01"   # Treatment starts Jan 2014

# States that made a clear legislative minimum wage increase in 2014
# (above the federal $7.25 floor they already exceeded in some cases,
#  or crossing above $7.25 for the first time)
TREATMENT_STATES = {
    "CT": "Connecticut",
    "NJ": "New Jersey",
    "NY": "New York",
    "CA": "California",
    "MN": "Minnesota",
    "MD": "Maryland",
    "MI": "Michigan",
    "HI": "Hawaii",
    "RI": "Rhode Island",
}

# States that stayed at federal minimum ($7.25) through entire window
CONTROL_STATES = {
    "TX": "Texas",
    "GA": "Georgia",
    "AL": "Alabama",
    "SC": "South Carolina",
    "NC": "North Carolina",
    "TN": "Tennessee",
    "MS": "Mississippi",
    "LA": "Louisiana",
    "IN": "Indiana",
    "UT": "Utah",
    "ID": "Idaho",
    "KS": "Kansas",
    "VA": "Virginia",
    "NH": "New Hampshire",
    "WY": "Wyoming",
}

# Minimum wage values for reference (annual, Jan of that year)
MIN_WAGE_INCREASES = {
    "CT": {"before": 8.25, "after": 8.70},
    "NJ": {"before": 7.25, "after": 8.25},
    "NY": {"before": 7.25, "after": 8.00},
    "CA": {"before": 8.00, "after": 9.00},
    "MN": {"before": 7.25, "after": 8.00},
    "MD": {"before": 7.25, "after": 8.00},
    "MI": {"before": 7.40, "after": 8.15},
    "HI": {"before": 7.25, "after": 7.75},
    "RI": {"before": 7.75, "after": 8.00},
}


def _get_fred() -> Fred:
    key = os.getenv("FRED_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets["fred"]["api_key"]
        except Exception:
            pass
    if not key:
        raise EnvironmentError(
            "FRED_API_KEY not set. Add it to .env or Streamlit secrets."
        )
    return Fred(api_key=key)


def fetch_unemployment(force_refresh: bool = False) -> pd.DataFrame:
    cache_path = CACHE_DIR / "unemployment.parquet"
    if cache_path.exists() and not force_refresh:
        return pd.read_parquet(cache_path)

    fred = _get_fred()
    all_states = {**TREATMENT_STATES, **CONTROL_STATES}
    frames = []

    for abbr, name in all_states.items():
        series_id = f"{abbr}UR"   # e.g. CTUR, NJUR, TXUR
        try:
            s = fred.get_series(series_id, observation_start=START, observation_end=END)
            s.name = abbr
            frames.append(s)
        except Exception as e:
            print(f"Warning: could not fetch {series_id} — {e}")

    df = pd.concat(frames, axis=1).dropna(how="all")
    df.index = pd.to_datetime(df.index)
    df.to_parquet(cache_path)
    return df


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide unemployment data to long panel format for DiD regression."""
    rows = []
    policy_ts = pd.Timestamp(POLICY_DATE)

    for abbr in df.columns:
        is_treated = abbr in TREATMENT_STATES
        for date, ur in df[abbr].dropna().items():
            rows.append({
                "state":      abbr,
                "date":       date,
                "year":       date.year,
                "month":      date.month,
                "treated":    int(is_treated),
                "post":       int(date >= policy_ts),
                "unemp_rate": ur,
            })

    panel = pd.DataFrame(rows).sort_values(["state", "date"]).reset_index(drop=True)
    panel["group"]  = panel["treated"].map({1: "Treatment (raised MW)", 0: "Control (held at $7.25)"})
    panel["period"] = panel["post"].map({1: "Post-2014", 0: "Pre-2014"})
    return panel
