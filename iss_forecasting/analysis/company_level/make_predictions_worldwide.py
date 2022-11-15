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
# Load model
model_path = PROJECT_DIR / "outputs/models/light_gbm_16h37m51s_2022-11-14"
model = load_pickle(model_path, "lightgbm_model.pickle")

# %%
# Load data
data = get_company_future_success_dataset(test=True, uk_only=False)

# %%
# Create dummy cols
data = pd.get_dummies(
    data,
    columns=["location_id", "last_investment_round_type"],
    prefix=["loc", "last_investment_round_type"],
)

# %%
# Remove non alphanumeric characters from column names
data = data.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))

# %%
# Make wordwide columns to drop
grants_cols = [col for col in data.columns if "grant" in col]
beis_cols = [col for col in data.columns if "beis" in col]
tech_cat_cols = [col for col in data.columns if "tech_cat_" in col]
other_cols_to_drop = ["id", "name", "legal_name", "long_description"]
cols_to_add_after_predictions = data[other_cols_to_drop]
worldwide_cols_to_drop = (
    grants_cols + beis_cols + tech_cat_cols + other_cols_to_drop + ["green_pilot"]
)

# %%
# Drop cols not used in the model
data = data.drop(columns=worldwide_cols_to_drop)

# %%
# Drop cols
data = data.drop(columns=["future_success"])

# %%
# Make predictions
preds_prob = model.predict_proba(data)[:, 1]
preds_binary = model.predict(data)

# %%
# Add predictions to data
data["success_pred_prob"] = preds_prob
data["success_pred_binary"] = preds_binary

# %%
data

# %%
# Add columns back in
data = pd.concat([data, cols_to_add_after_predictions], axis=1)

# %%
data

# %%
# Drop dummy cols
loc_data = data.filter(regex="^loc_")
lirt_data = data.filter(regex="^last_investment_round_type_")
data = data.drop(columns=loc_data.columns).drop(columns=lirt_data.columns)

# %%
data

# %%
# Add useful cols from crunchbase orgs file
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

predictions = data[["id", "success_pred_prob", "success_pred_binary"]].merge(
    right=cb_orgs, on="id"
)

# %%
# Save all companies worldwide
(
    predictions.sort_values("success_pred_prob", ascending=False)
    .reset_index(drop=True)
    .to_csv(PROJECT_DIR / "outputs/data/all_worldwide_companies_success_preds.csv")
)

# %%
# Filter for dealroom companies only and save that file

# %%
