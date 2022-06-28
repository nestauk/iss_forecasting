"""Data getters for company future success prediction"""
import pandas as pd
from iss_forecasting import PROJECT_DIR


def get_company_future_success_dataset(
    date_range: str = "2011-01-01-2019-01-01",
) -> pd.DataFrame:
    """Load company level future success dataset

    Args:
        date_range: Date range in the format - 2011-01-01-2019-01-01

    Returns:
        Dataset of companies including information about investments,
        grants and future success
    """
    return pd.read_csv(
        PROJECT_DIR / f"inputs/data/company_level/company_data_window_{date_range}.csv",
        index_col=0,
    )
