"""Processing utils
"""
import pandas as pd


def find_zero_items(df: pd.DataFrame, item_col: str, value_col: str) -> list:
    """Returns a list of items with 0s for all records in a
    specified value column

    Args:
        df: Dataframe to find items with 0 values
        item_col: Column with item names
        value_col: Column with value to check for 0s

    Returns:
        List of items with 0s for all records in a specified value column
    """
    return [
        item
        for item in df[item_col].unique()
        if df.query(f"{item_col} == '{item}'")[value_col].sum() == 0
    ]
