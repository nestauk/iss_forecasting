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

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# %%
# Load model
model_path = PROJECT_DIR / "outputs/models/light_gbm_21h25m19s_2022-04-27"
model = load_pickle(model_path, "lightgbm_model.pickle")

# %%
# Load dataset
data = get_company_future_success_dataset(date_range="2014-01-04-2022-01-04")

# %%
# Process dataset
cols_not_used_by_model = [
    "id",
    "name",
    "legal_name",
    "long_description",
    "future_success",
    "location_id",
    "last_investment_round_type",
]
cols_to_add_after_predictions = data[cols_not_used_by_model].drop(
    columns="future_success"
)
dummies = pd.get_dummies(
    data[["location_id", "last_investment_round_type"]],
    columns=["location_id", "last_investment_round_type"],
    prefix=["loc", "last_investment_round_type"],
)

data = pd.concat([data, dummies], axis=1).drop(columns=cols_not_used_by_model)

data.columns = data.columns.str.replace("beis_", "")

# %%
# Load training dataset
X_train, _, _, _ = load_pickle(model_path, "datasets.pickle")

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

# %%
# Add missing cols with appropriate value
for col in cols_to_set_to_zero:
    data[col] = 0
for col in cols_to_set_to_neg_one:
    data[col] = -1

# %%
# Make predictions
preds_prob = model.predict_proba(data)[:, 1]
preds_binary = model.predict(data)

# %%
# Add predictions to data
data["success_pred_prob"] = preds_prob
data["success_pred_binary"] = preds_binary

# %%
# Add columns back in
data = pd.concat([data, cols_to_add_after_predictions], axis=1)

# %%
# Drop cols with no data
data = data.drop(columns=cols_to_add)

# %%
# Drop dummy cols
loc_data = data.filter(regex="^loc_")
lirt_data = data.filter(regex="^last_investment_round_type_")
data = data.drop(columns=loc_data.columns).drop(columns=lirt_data.columns)

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

predictions = data[
    ["id", "success_pred_prob", "success_pred_binary", "green_pilot"]
].merge(right=cb_orgs, on="id")

# %%
# Save all UK companies
(
    predictions.drop(columns="green_pilot")
    .sort_values("success_pred_prob", ascending=False)
    .reset_index(drop=True)
    .to_csv(PROJECT_DIR / "outputs/data/all_uk_companies_success_preds.csv")
)

# %%
# Find and save top 20 most confident success predictions for green companies
(
    predictions.query("green_pilot == 1")
    .sort_values("success_pred_prob", ascending=False)[0:20]
    .to_csv(PROJECT_DIR / "outputs/data/top_20_green_success_preds.csv")
)
