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
import lightgbm as lgb
from iss_forecasting import PROJECT_DIR, logger
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import pathlib
from pathlib import Path
from tqdm import tqdm
from iss_forecasting.getters.company_success import get_company_future_success_dataset
from iss_forecasting.utils.io import save_pickle
import re


# %%
def drop_cols(cfs_data: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not needed in the worldwide prediction model"""
    return cfs_data.drop(
        columns=[
            col
            for col in cfs_data.columns
            if "grant" in col or "beis" in col or "tech_cat_" in col
        ]
        + ["green_pilot", "id", "legal_name", "long_description", "name"]
    )


def make_dummies(cfs_data: pd.DataFrame) -> pd.DataFrame:
    """Make dummy columns for location_id and last_investment_round_type"""
    return pd.get_dummies(
        cfs_data,
        columns=["location_id", "last_investment_round_type"],
        prefix=["loc", "last_investment_round_type"],
    )


def set_names_variable(cfs_data: pd.DataFrame) -> pd.DataFrame:
    """Create global variable for names for the companies"""
    global names
    names = cfs_data.name
    return cfs_data


def train_valid_split(cfs_data: pd.DataFrame, val_split: float) -> pd.DataFrame:
    """Split the training data into X_train, y_train, X_valid, y_valid
    Also return a global variable for valid_names which can be used as index
    in the explainder dashboard.

    Args:
        cfs_data: Company future success dataset
        val_split: Size of the validation dataset e.g 0.2

    Returns:
        X_train, y_train, X_valid, y_valid
    """
    val_size = int(len(cfs_data) * val_split)
    train_data = cfs_data.head(-val_size)
    validation_data = cfs_data.tail(val_size)
    global valid_names
    valid_names = names.tail(val_size)  # To be used as index in explainer dashboard
    X_train = train_data.drop(columns=["future_success"])
    y_train = train_data.future_success
    X_valid = validation_data.drop(columns=["future_success"])
    y_valid = validation_data.future_success
    logger.info(f"Training data mean future success is: {y_train.mean()}")
    logger.info(f"Validation data mean future success is: {y_valid.mean()}")
    return X_train, y_train, X_valid, y_valid


# %%
# Load data
dataset_size = 180_000  # Reduce if having RAM issues
data = (
    get_company_future_success_dataset(test=False, uk_only=False)
    .sample(frac=1, random_state=10)
    .head(dataset_size)
    .pipe(set_names_variable)
    .pipe(drop_cols)
    .assign(
        location_id=lambda x: x.location_id.str.replace(
            "[^A-Za-z0-9_]+", "", regex=True
        )
    )
    .pipe(make_dummies)
)

# %%
# Split data
X_train, y_train, X_valid, y_valid = train_valid_split(data, val_split=0.2)
del data, names

# %%
# Create new directory to save model to
dt = time.strftime("%Hh%Mm%Ss_%Y-%m-%d")
model_path = PROJECT_DIR / f"outputs/models/light_gbm_{dt}_{dataset_size}"
model_path.mkdir(parents=True, exist_ok=True)

# %%
# # Create dict of parameters to search
# # Note with the below search params, it will take ~1 hour to train on an M1 mac
# search_params = {
#     "num_leaves": [32, 35, 48],
#     "max_depth": [11, 12, 13],
#     "scale_pos_weight": [2.5, 2.55],
#     "min_child_weight": [0.002, 0.003],
#     "subsample": [0.4, 1],
#     "colsample_bytree": [0.81, 0.83],
# }

# # Create lists to record results
# iteration = []
# val_log_loss = []
# val_f1 = []
# params = []

# # Search all combinations of parameters
# for i, p in tqdm(enumerate(ParameterGrid(search_params))):
#     model = lgb.LGBMClassifier(
#         boosting_type="gbdt",
#         num_leaves=31,
#         max_depth=-1,
#         learning_rate=0.05,
#         n_estimators=100_000,
#         subsample_for_bin=200_000,
#         objective="binary",
#         class_weight=None,
#         min_split_gain=0.0,
#         min_child_weight=0.001,
#         min_child_samples=20,
#         subsample=1.0,
#         subsample_freq=0,
#         colsample_bytree=1.0,
#         reg_alpha=0.0,
#         reg_lambda=0.0,
#         random_state=None,
#         n_jobs=-1,
#         importance_type="split",
#         scale_pos_weight=2.3,
#     )
#     model.set_params(**p)
#     eval_results = {}
#     model.fit(
#         X=X_train,
#         y=y_train,
#         callbacks=[
#             lgb.early_stopping(stopping_rounds=40),
#             lgb.record_evaluation(eval_results),
#         ],
#         eval_set=[(X_train, y_train), (X_valid, y_valid)],
#         eval_metric="logloss",
#     )
#     val_log_loss.append(min(eval_results["valid_1"]["binary_logloss"]))
#     val_f1.append(f1_score(y_valid, model.predict(X_valid)))
#     iteration.append(i)
#     params.append(model.get_params())

# search_results = {
#     "iteration": iteration,
#     "params": params,
#     "val_f1": val_f1,
#     "val_logloss": val_log_loss,
# }
# search_results_df = pd.DataFrame.from_dict(search_results)

# %%

# %%
# # Save parameter search results
# search_results_df.to_csv(model_path / "param_search.csv", index=False)

# # Save best validation F1 score parameters
# iterations_max_f1 = search_results_df[
#     search_results_df["val_f1"] == search_results_df["val_f1"].max()
# ]
# max_val_f1 = iterations_max_f1.params.values[0]
# save_pickle(model_path, "max_val_f1", max_val_f1)

# %%
# Save X_train, y_train, X_valid, y_valid
save_pickle(model_path, "datasets", (X_train, y_train, X_valid, y_valid))

# Save valid_names
save_pickle(model_path, "valid_names", valid_names)

# %%
# Use this cell if you want to run without doing parameter search
import ast

search_results_df = pd.read_csv(
    PROJECT_DIR / "outputs/models/light_gbm_21h25m19s_2022-04-27/param_search.csv"
)
iterations_max_f1 = search_results_df[
    search_results_df["val_f1"] == search_results_df["val_f1"].max()
]
max_val_f1 = ast.literal_eval(iterations_max_f1.params.values[0])

# %%
# Settings to reduce RAM usage
max_val_f1["num_leaves"] = 10
max_val_f1["histogram_pool_size"] = 1024
max_val_f1["max_bins"] = 128

# %%
# Train model with best validation F1 score parameters
# Note this training only takes seconds to run
evals_result = {}

model = lgb.LGBMClassifier()
model.set_params(**max_val_f1)

model.fit(
    X=X_train,
    y=y_train,
    callbacks=[
        lgb.early_stopping(stopping_rounds=40),
        lgb.record_evaluation(evals_result),
    ],
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_metric="logloss",
)
# Save model
save_pickle(model_path, "lightgbm_model", model)

# Save training loss val vs train over iterations
lgb.plot_metric(evals_result, metric="binary_logloss")
plt.savefig(model_path / "training_loss.jpeg")

# Save confusions matrix
y_true = y_valid.values
y_pred = model.predict(X_valid)
cm = confusion_matrix(y_true, y_pred)
tpr = round(cm[1][1] / (cm[1][1] + cm[1][0]), 3)
tnr = round(cm[0][0] / (cm[0][0] + cm[0][1]), 3)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["no future success", "future success"]
)
disp.plot()
disp.ax_.set(xlabel="Predicted", ylabel="Actual")
plt.title(f"Confusion matrix - validation data. TPR = {tpr}, TNR = {tnr}")
plt.tight_layout()
plt.savefig(model_path / "valid_confusion_matrix.jpeg")

# Save X_train, y_train, X_valid, y_valid
save_pickle(model_path, "datasets", (X_train, y_train, X_valid, y_valid))

# Save valid_names
save_pickle(model_path, "valid_names", valid_names)

# %%
