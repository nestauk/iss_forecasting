# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Estimating causal effects relating to future company success
# This notebook uses the library [DoWhy](https://microsoft.github.io/dowhy/index.html) and their 4 step process to calculate causal effects relating to future company success.

# %% [markdown]
# ## Step 0: Import libraries and load data

# %%
from iss_forecasting import PROJECT_DIR
from iss_forecasting.analysis.company_level.causal_graphs import (
    small_graph,
    medium_graph,
    large_graph,
)
from iss_forecasting.getters.company_success import get_company_future_success_dataset
from dowhy import CausalModel
import dowhy.datasets
import pygraphviz
from IPython.display import Image, display
import warnings
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

# %%
CAUSAL_GRAPH_IMG_PATH_SAVE = PROJECT_DIR / "outputs/figures/causal_model"
CAUSAL_GRAPH_IMG_PATH_LOAD = PROJECT_DIR / "outputs/figures/causal_model.png"

# %%
data = get_company_future_success_dataset()

# %% [markdown]
# ## Step 1: Create causal graph
# The first step is to create a causal graph. There are three causal graphs below:
# - small_graph: graph containing only `founder_mean_degrees`, `total_investment_amount_gbp`, `has_received_ukri_grant`, `has_received_grant`, `future_success`
# - large_graph: graph as designed in [Miro board](https://miro.com/app/board/uXjVO9Mipv4=/?share_link_id=440404467074)
# - medium_graph: same as large graph but without industry and beis indicator related nodes (use medium to view the graph below as the large graph has too many nodes to be visualised)
#
# Note that when using the large graph, the calculations in steps 2 and 4 can take over an hour. Also, as the large graph contains so many nodes, it is hard to visualise the graph.
#
# Uncomment the graph and treatment below that you would like to use.

# %%
# causal_graph = small_graph()
causal_graph = medium_graph()
# causal_graph = large_graph(data)

# %%
treatment = "has_received_grant"
# treatment = "has_received_ukri_grant"

# %%
model = dowhy.CausalModel(
    data=data,
    graph=causal_graph,
    treatment=treatment,
    outcome="future_success",
)
model.view_model(file_name=CAUSAL_GRAPH_IMG_PATH_SAVE, size=(24, 18))
display(Image(filename=CAUSAL_GRAPH_IMG_PATH_LOAD))

# %% [markdown]
# ## Step 2: Use the graph to identify a target estimand
# Note that this step is computed in seconds using the small and medium causal graphs but the large graph takes ~1hr 15mins on an M1 mac.

# %%
# %%time
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)

# %% [markdown]
# ## Step 3: Estimate the causal effect based on the identified estimand
# Estimate the average treatment effect effect using propensity score weighting.

# %%
# %%time
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_weighting",
    target_units="ate",
    # confidence_intervals=True  # Calculating the ci takes a long time
)
print(estimate)

# %%
interpretation = estimate.interpret(method_name="textual_effect_interpreter")

# %% [markdown]
# ## Step 4: Refute the obtained estimate
# Note that these refutation calculations take hours when using the full graph.

# %% [markdown]
# Does the estimated effect change when a common cause independent random variable is added? It should stay the same.

# %%
# %%time
refute1_results = model.refute_estimate(
    identified_estimand, estimate, method_name="random_common_cause"
)
print(refute1_results)

# %% [markdown]
# Does the estimated effect change when we change the true treatment variable with an independent random variable? The estimated effect should go to zero.

# %%
# %%time
refute2_results = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="placebo_treatment_refuter",
    placebo_type="permute",
)
print(refute2_results)

# %% [markdown]
# Does the estimated effect change when we replace the given dataset with a randomly selected subset? The estimated effect should change slightly.

# %%
# %%time
refute3_results = model.refute_estimate(
    identified_estimand, estimate, method_name="data_subset_refuter"
)
print(refute3_results)
