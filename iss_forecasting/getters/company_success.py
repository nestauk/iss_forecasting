"""Data getters for company future success prediction"""
import pandas as pd
from iss_forecasting import PROJECT_DIR


def get_company_future_success_dataset() -> pd.DataFrame:
    """Load company level future success dataset"""
    return pd.read_csv(
        PROJECT_DIR
        / "inputs/data/company_level/company_data_window_2011-01-01-2019-01-01.csv",
        index_col=0,
    )
