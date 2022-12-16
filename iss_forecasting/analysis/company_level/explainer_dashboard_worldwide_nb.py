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
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from iss_forecasting import PROJECT_DIR
from iss_forecasting.utils.io import load_pickle

# %%
model_path = PROJECT_DIR / "outputs/models/light_gbm_10h53m55s_2022-11-21_180000"

# %%
# Load objects needed for explainer dashboard
model = load_pickle(model_path, "lightgbm_model.pickle")
X_train, y_train, X_valid, y_valid = load_pickle(model_path, "datasets.pickle")
valid_names = load_pickle(model_path, "valid_names.pickle")

# %%
explainer = ClassifierExplainer(
    model,
    X_valid,
    y_valid,
    X_background=X_train,
    cats=["loc", "last_investment_round_type", "group"],
    labels=["No Future Success", "Future Success"],
    na_fill=-1,
    idxs=valid_names,
    index_name="Company name",
)

db = ExplainerDashboard(
    explainer,
    title="Company Success Explainer",
    simple=False,
    whatif=True,
    shap_interaction=False,
    decision_trees=False,
    mode="external",
)
db.run(port=8055)
