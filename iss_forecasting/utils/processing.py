"""Processing utils
"""


def find_zero_items(df, item_col, value_col):
    """Returns a list of items with 0s for all records in a
    specified value column

    Args:
        df (df): dataframe to find items with 0 values
        item_col (str): column with item names
        value_col (str): column with value to check for 0s

    Returns:
        list: list of items with 0s for all records in a specified value column
    """
    return [
        item
        for item in df[item_col].unique()
        if df.query(f"{item_col} == '{item}'")[value_col].sum() == 0
    ]
