"""Data getters for company future success prediction"""
import pandas as pd
from iss_forecasting import PROJECT_DIR


def get_company_future_success_dataset(
    date_range: str = "2011-01-01-2019-01-01", uk_only: bool = True, test: bool = False
) -> pd.DataFrame:
    """Load company level future success dataset

    Args:
        date_range: Date range in the format - 2011-01-01-2019-01-01
        uk_only: True to load a uk only data, False to load a worldwide data
        test: True to load a test dataset, False to load a full dataset

    Returns:
        Dataset of companies including information about investments,
        grants and future success
    """
    test_indicator = "_test" if test else ""
    region_indicator = "_ukonly" if uk_only else "_worldwide"
    return pd.read_csv(
        PROJECT_DIR
        / f"inputs/data/company_level/company_data_window_{date_range}{test_indicator}{region_indicator}.csv",
        index_col=0,
    )
