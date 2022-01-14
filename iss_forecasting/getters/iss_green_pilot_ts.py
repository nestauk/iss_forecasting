"""Data getters for the ISS green pilot time series data.
Data source: https://github.com/nestauk/innovation_sweet_spots
"""
import pandas as pd
from iss_forecasting import PROJECT_DIR


def get_iss_green_pilot_time_series() -> pd.DataFrame:
    """Load ISS green pilot time series data,
    rename columns to more be more descriptive
    """
    return pd.read_csv(PROJECT_DIR / "inputs/data/ISS_pilot_Time_series.csv").rename(
        columns={
            "no_of_projects": "research_no_of_projects",
            "amount_total": "research_funding_total",
            "raised_amount_gbp_total": "investment_raised_total",
            "no_of_rounds": "investment_rounds",
        }
    )
