# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from iss_forecasting import PROJECT_DIR
from iss_forecasting.utils.io import load_pickle
from iss_forecasting.getters.company_success import get_company_future_success_dataset
from iss_forecasting.getters.crunchbase import get_crunchbase_orgs
import pandas as pd
import warnings
import re

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# %%
# Dataset split value
split = 1

# %%
# Load model
model_path = PROJECT_DIR / "outputs/models/light_gbm_10h53m55s_2022-11-21_180000"
model = load_pickle(model_path, "lightgbm_model.pickle")
# Load data
data = get_company_future_success_dataset(
    date_range="2014-01-04-2022-01-04", test=False, uk_only=False, split=split
)
# Load training dataset
X_train, _, _, _ = load_pickle(model_path, "datasets.pickle")

# %%
# Create dummy cols
data = pd.get_dummies(
    data,
    columns=["location_id", "last_investment_round_type"],
    prefix=["loc", "last_investment_round_type"],
)
# Remove non alphanumeric characters from column names
data = data.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
# Make wordwide columns to drop
other_cols_to_drop = ["id", "name", "legal_name", "long_description"]
cols_to_add_after_predictions = data[other_cols_to_drop]
worldwide_cols_to_drop = (
    [
        col
        for col in data.columns
        if "grant" in col or "beis" in col or "tech_cat_" in col
    ]
    + other_cols_to_drop
    + ["green_pilot"]
)
# Drop cols not used in the model
data = data.drop(columns=worldwide_cols_to_drop)
# Drop future success
data = data.drop(columns=["future_success"])

# %%
# Calculate missing columns from data that will need to be added so it can be used in the model
cols_to_add = set(X_train.columns) - set(data.columns)
cols_to_drop = set(data.columns) - set(X_train.columns)

# %%
# Drop cols
data = data.drop(columns=cols_to_drop)

# %%
# Calculate which cols need to be set to 0 or -1
cols_to_set_to_zero = [
    col for col in cols_to_add if col[:4] == "loc_" or col[:6] == "group_"
]
cols_to_set_to_neg_one = [
    col for col in cols_to_add if col[:4] != "loc_" and col[:6] != "group_"
]

# Add missing cols with appropriate value
for col in cols_to_set_to_zero:
    data[col] = 0
for col in cols_to_set_to_neg_one:
    data[col] = -1

# %%
# Remove undeed variables to avoid running out of RAM
del X_train, cols_to_drop, _, cols_to_add, cols_to_set_to_zero

# %%
# Remove duplicate columns
data = data.loc[:, ~data.columns.duplicated()]


# %%
def add_predictions(data: pd.DataFrame, model) -> pd.DataFrame:
    pred_prob = model.predict_proba(data)[:, 1]
    pred_binary = model.predict(data)
    return data.assign(success_pred_prob=pred_prob, success_pred_binary=pred_binary)


def drop_dummy_cols(data: pd.DataFrame) -> pd.DataFrame:
    loc_data = data.filter(regex="^loc_")
    lirt_data = data.filter(regex="^last_investment_round_type_")
    return data.drop(columns=loc_data.columns).drop(columns=lirt_data.columns)


def add_info_cols_back(
    data: pd.DataFrame, cols_to_add_after_predictions: list
) -> pd.DataFrame:
    return pd.concat([data, cols_to_add_after_predictions], axis=1)


def add_cb_cols(data: pd.DataFrame) -> pd.DataFrame:
    cb_orgs = get_crunchbase_orgs()[
        [
            "id",
            "name",
            "legal_name",
            "short_description",
            "long_description",
            "founded_on",
            "location_id",
            "employee_count",
            "num_funding_rounds",
            "total_funding_usd",
            "homepage_url",
            "email",
            "cb_url",
        ]
    ]
    return data[["id", "success_pred_prob", "success_pred_binary"]].merge(
        right=cb_orgs, on="id"
    )


# %%
# Make predictions
data = (
    data.pipe(add_predictions, model)
    .pipe(drop_dummy_cols)
    .pipe(add_info_cols_back, cols_to_add_after_predictions)
    .pipe(add_cb_cols)
)

# %%
# Save worldwide companies with predictions
data.to_csv(
    PROJECT_DIR
    / f"outputs/data/all_worldwide_companies_success_preds_split_{split}.csv"
)

# %%
# Once predictions have run for each split, the files can be combined,
# dealroom ids added and then saved to csv
split_1 = pd.read_csv(
    PROJECT_DIR / "outputs/data/all_worldwide_companies_success_preds_split_1.csv",
    index_col=0,
)
split_2 = pd.read_csv(
    PROJECT_DIR / "outputs/data/all_worldwide_companies_success_preds_split_2.csv",
    index_col=0,
)
split_3 = pd.read_csv(
    PROJECT_DIR / "outputs/data/all_worldwide_companies_success_preds_split_3.csv",
    index_col=0,
)
dr_cb_link = pd.read_csv(
    PROJECT_DIR / "inputs/data/dr_cb_link/dr_cb_lookup.csv", index_col=0
)
(
    pd.concat([split_1, split_2, split_3])
    .merge(dr_cb_link, left_on="id", right_on="id_cb", how="left")
    .sort_values("success_pred_prob", ascending=False)
    .drop(columns="id_cb")
    .to_csv(PROJECT_DIR / f"outputs/data/all_worldwide_companies_success_preds.csv")
)
