"""Data getters for the ISS green pilot time series data.
Data source: https://github.com/nestauk/innovation_sweet_spots
"""
import pandas as pd
import os
from iss_forecasting import PROJECT_DIR


def get_iss_green_gtr_ts(split: bool, period: str) -> pd.DataFrame:
    """Load ISS green pilot gateway to research data,
    rename columns to be more descriptive

    Args:
        split: True if want to load data where research funding amounts
            has been split across the duration of the research project.
            False if want to load data where research funding amount
            has been attributed to the start date of the research project.
        period: Time period that the data has been grouped by,
            'year', 'month', 'quarter'.

    Returns:
        Dataframe with columns:
            - time_period
            - research_no_of_projects
            - research_funding_total
            - tech_category
    """
    path = (
        PROJECT_DIR / f"inputs/data/gtr_split/{period}/"
        if split
        else PROJECT_DIR / f"inputs/data/gtr_not_split/{period}/"
    )
    return (
        pd.concat(
            [
                pd.read_csv(path / file)
                for file in os.listdir(path)
                if file.endswith(".csv")
            ],
            ignore_index=True,
        )
        .rename(
            columns={
                "no_of_projects": "research_no_of_projects",
                "amount_total": "research_funding_total",
            }
        )
        .astype({"time_period": "datetime64[ns]"})
    )


def get_iss_green_cb_ts(period: str) -> pd.DataFrame:
    """Load ISS green pilot crunchbase data,
    rename columns to be more descriptive

    Args:
        period: Time period that the data has been grouped by,
            'year', 'month', 'quarter'.

    Returns:
        Dataframe with columns:
            - time_period
            - investment_rounds
            - investment_raised_total
            - no_of_orgs_founded
            - tech_category
    """
    path = PROJECT_DIR / f"inputs/data/cb/{period}/"
    return (
        pd.concat(
            [
                pd.read_csv(path / file)
                for file in os.listdir(path)
                if file.endswith(".csv")
            ],
            ignore_index=True,
        )
        .rename(
            columns={
                "raised_amount_gbp_total": "investment_raised_total",
                "no_of_rounds": "investment_rounds",
            }
        )
        .astype({"time_period": "datetime64[ns]"})
        .drop(columns=["raised_amount_usd_total"])
    )
