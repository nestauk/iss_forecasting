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
# # Exploring the relationship between research funding and private investment
# This notebook looks at the relationship between research funding and private investment. It uses pilot data produced from [Innovation Sweet Spots](https://github.com/nestauk/innovation_sweet_spots) that focuses on companies in technology sectors that could help tackle climate change. The underlying research funding data comes from UKRI's Gateway to Research and private investment data comes from Crunchbase.
# %%
from iss_forecasting.getters.iss_green_pilot_ts import (
    get_iss_green_gtr_ts,
    get_iss_green_cb_ts,
)
from iss_forecasting.utils.processing import find_zero_items
from iss_forecasting.analysis.utils.plotting import plot_two_y_one_x, lagplot, plot_lags
from iss_forecasting.utils.stats_tests import stationarity_adf, stationarity_kpss
import altair as alt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import ccf
from scipy.signal import detrend
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
from statsmodels.tsa.stattools import InterpolationWarning

warnings.filterwarnings(action="ignore", category=InterpolationWarning)

# %%
# load iss green time series yearly data where research funding is attributed to the start of the project
iss_gtr_ts_yearly = get_iss_green_gtr_ts(split=True, period="quarter")
iss_cb_ts_yearly = get_iss_green_cb_ts(period="quarter")
iss_ts = pd.merge(
    left=iss_gtr_ts_yearly, right=iss_cb_ts_yearly, on=["time_period", "tech_category"]
)

# %%
# view sample of dataframe
iss_ts.tail(5)

# %% [markdown]
# Column descriptions:<br>
# `time_period` = time period in YYYY-MM-DD format<br>
# `research_no_of_projects` = number of research projects started in a specific `year` (GtR)<br>
# `research_funding_total` = research funding amount (GBP 1000s) awarded in a specific `year` (GtR)<br>
# `investment_rounds` = number of rounds of investment in a specific `year` (Crunchbase)<br>
# `investment_raised_total` = raised investment amount (GBP 1000s) in a specific `year` (Crunchbase)<br>
# `no_of_orgs_founded` = number of new companies started in the `year` (Crunchbase)<br>
# `tech_category` = technology category e.g 'Heat pumps', 'Batteries'

# %%
# see tech categories
iss_ts.tech_category.unique()

# %% [markdown]
# Note that some of the technology categories include other tech categories.<br>
# `'Low carbon heating'` includes:
# - `'Heat pumps'`
# - `'Geothermal energy'`
# - `'Solar thermal'`
# - `'District heating'`
# - `'Hydrogen heating'`
# - `'Biomass heating'`
# - `'Micro CHP'`
# - `'Heat storage'`
#
# `'EEM'` (energy efficiency and management) includes:
# - `'Insulation & retrofit'`
# - `'Energy management'` -- this includes companies working on things like smart homes, smart meters, demand response and load shifting
#
# Note that some companies in this dataset can belong to more than one technology category.
# %%
# check for tech categories with no investment
categories_with_no_investment = find_zero_items(
    df=iss_ts, item_col="tech_category", value_col="investment_raised_total"
)
categories_with_no_investment
# %%
# remove tech categories with no investment from time series data
iss_ts = iss_ts.query(f"tech_category != {categories_with_no_investment}").reset_index(
    drop=True
)
# %%
# see remaining tech categories
iss_ts.tech_category.unique()
# %% [markdown]
# ## Plot research funding vs private investment for each tech category

# %%
tech_cats = iss_ts.tech_category.unique()
plots_ts = alt.vconcat()
for tech_cat in tech_cats:
    iss_ts_current_tc = iss_ts.query(f"tech_category == '{tech_cat}'")
    plots_ts &= plot_two_y_one_x(
        data_source=iss_ts_current_tc,
        x="time_period:T",
        y1="research_funding_total:Q",
        y2="investment_raised_total:Q",
        chart_title=f"{tech_cat} research funding vs. investment over time",
    )

plots_ts

# %% [markdown]
# Looking at the above plots, there does not seem to be an obvious relationship between research funding and private investment that is consistent across all tech categories.

# %% [markdown]
# ## Cross Correlation Plots
# Plot research funding lagged against private investment for each technology category

# %%
for tech_cat in iss_ts.tech_category.unique():
    iss_ts_current_tc = iss_ts.query(f"tech_category == '{tech_cat}'")
    plot = plot_lags(
        x=iss_ts_current_tc.research_funding_total,
        title=tech_cat,
        y=iss_ts_current_tc.investment_raised_total,
        lags=20,
        nrows=3,
    )

# %% [markdown]
# Looking at the above lag plots, there does not seem to be a common time lag between research funding and private investment.<br>
# `Micro CHP`,`Solar thermal`, `Insulation & retrofit`, `Solar`, `Wind & offshore`, `Hydrogen & fuel cells`, `Bioenergy`, `Carbon capture & storage`  -- there are no significant correlations at any lag.<br>
# `Heat Storage` -- there are correlations at lags 0, 2, and 4. <br>
# `Heat Pumps` -- there is correlation at lag 3.<br>
# `Hydrogen heating` -- there is correlation at lag 2.<br>
# `Biomass heating` -- there is correlation at lag 1.<br>
# `Energy management` -- there is correlation at lag 4 and lag 6.<br>
# `Low carbon heating` -- there is correlation at lag 0 and negative correlation at lag 4 and lag 9.<br>
# `EEM` -- there is correlation at lags 4 and 6.<br>
# `Batteries` -- the correlations are significant at all lags, with strong correlations from lags 1 and higher.<br>

# %% [markdown]
# # Granger Causality
# The Granger causality test is a statistical hypothesis test for determining whether one time series is useful in forecasting another. The two time series being tested need to be stationary. We can test for stationarity using the ADF and KPSS tests.<br><br>
# From [statsmodels stationarity and detrending example](https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html):<br>
# It is always better to apply both the ADF and KPSS tests, so that it can be ensured that the series is truly stationary. Possible outcomes of applying these stationary tests are as follows:
#
# * Case 1: Both tests conclude that the series is not stationary - The series is not stationary
# * Case 2: Both tests conclude that the series is stationary - The series is stationary
# * Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.
# * Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. Differencing is to be used to make series stationary. The differenced series is checked for stationarity.

# %% [markdown]
# We will test for granger causality in the battery industry.

# %%
tech_cat = "Energy management"
tc_ts = iss_ts[iss_ts.tech_category == tech_cat]
tc_ts_rf = tc_ts.research_funding_total
tc_ts_pi = tc_ts.investment_raised_total

# %% [markdown]
# First check if the time series are stationary:

# %%
stationarity_adf(tc_ts_rf, f"{tech_cat} research funding")
stationarity_kpss(tc_ts_rf, f"{tech_cat} research funding")
stationarity_adf(tc_ts_pi, f"{tech_cat} investment")
stationarity_kpss(tc_ts_pi, f"{tech_cat} investment")

# %% [markdown]
# Batteries research funding is stationary according to KPSS but not stationary according to ADF, to try and remove stationarity from the time series, we can detrend the time series.<br>
# Batteries private investment is not stationary by KPSS or ADF, to try and remove stationarity from the time series, we can difference the time series.

# %%
tc_ts_rf_diff = tc_ts_rf.diff().dropna()
tc_ts_pi_diff = tc_ts_pi.diff().dropna()

# %% [markdown]
# Check again if the time series are stationary:

# %%
stationarity_adf(tc_ts_rf_diff, f"{tech_cat} research funding")
stationarity_kpss(tc_ts_rf_diff, f"{tech_cat} research funding")
stationarity_adf(tc_ts_pi_diff, f"{tech_cat} investment")
stationarity_kpss(tc_ts_pi_diff, f"{tech_cat} investment")

# %% [markdown]
# The research funding ADF test is stationary when the detrending is done with breakpoints at 0th and 10th time points (`bp=[0, 10]`).<br>
# Try rerunning the detrending with `bp=[0]` to see that the research funding ADF test still says the time series is not stationary (although much closer to being stationary than before).

# %%
combine_ts = pd.concat([tc_ts_rf_diff, tc_ts_pi_diff], axis=1)
combine_ts = combine_ts.tail(-1)

# %% [markdown]
# Due to the differencing, investment_rasied_total doesn't have a value for the first row. So the first row can be dropped.

# %% [markdown]
# The granger causality test used below tests for if the second time series is a predictor for the first time series. More details about the function can be found [here](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.grangercausalitytests.html). If all 4 tests p values are below 0.05, we can say the second time series granger causes the first time series.

# %%
rf_predict_pi = grangercausalitytests(
    combine_ts[["investment_raised_total", "research_funding_total"]], maxlag=3
)

# %% [markdown]
# This suggests that research funding lagged by 4 years (but not lagged by 1-3 years) is a predictor for private investment.

# %%
pi_predict_rf = grangercausalitytests(
    combine_ts[["research_funding_total", "investment_raised_total"]], maxlag=3
)

# %% [markdown]
# This suggests that private investment (lagged from 1-4 years) is not a predictor for research funding.

# %% [markdown]
# At this stage, I don't think we can draw too much from this but this process can be rerun when we have the data at split at smaller time intervals.

# %%
import pmdarima as pm
from pmdarima import model_selection
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# %%
tech_cat = "Batteries"
tc_ts = iss_ts[iss_ts.tech_category == tech_cat]
tc_ts_rf = tc_ts.research_funding_total
tc_ts_pi = tc_ts.investment_raised_total

# %%
test_periods = 4
tc_ts_pi_train = tc_ts_pi.head(-4)
tc_ts_pi_test = tc_ts_pi.tail(4)
tc_ts_rf_train = tc_ts_rf.head(-4)
tc_ts_rf_test = tc_ts_rf.tail(4)

# %%
modl = pm.auto_arima(
    tc_ts_pi_train,
    X=tc_ts_rf_train.values.reshape(-1, 1),
    start_p=1,
    start_q=1,
    start_P=1,
    start_Q=1,
    max_p=5,
    max_q=5,
    max_P=5,
    max_Q=5,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    D=10,
    max_D=10,
    error_action="ignore",
)

# %%
modl

# %%
tc_ts_pi_test.shape[0]

# %%
# Create predictions for the future, evaluate on test
preds, conf_int = modl.predict(
    n_periods=tc_ts_pi_test.shape[0],
    X=tc_ts_rf_test.values.reshape(-1, 1),
    return_conf_int=True,
)

# %%
preds

# %%
conf_int

# %%

# %%
# Create predictions for the future, evaluate on test
preds, conf_int = modl.predict(
    n_periods=tc_ts_pi_test.shape[0],
    X=tc_ts_rf_test.values.reshape(-1, 1),
    return_conf_int=True,
)

# Print the error:
print("Test RMSE: %.3f" % np.sqrt(mean_squared_error(tc_ts_pi_test, preds)))

# %%
from merlion.utils import TimeSeries
from ts_datasets.forecast import M4

# Data loader returns pandas DataFrames, which we convert to Merlion TimeSeries
time_series, metadata = M4(subset="Yearly")[0]
train_data = TimeSeries.from_pd(time_series[metadata.trainval])
test_data = TimeSeries.from_pd(time_series[~metadata.trainval])

# %%

# %%
tech_cat = "Batteries"
tc_ts = iss_ts[iss_ts.tech_category == tech_cat]
test_periods = 4
tc_ts_rf = tc_ts[["time_period", "research_funding_total"]].set_index("time_period")
tc_ts_pi = tc_ts[["time_period", "investment_raised_total"]].set_index("time_period")
tc_ts_pi_train = tc_ts_pi.head(-test_periods)
tc_ts_pi_test = tc_ts_pi.tail(test_periods)
tc_ts_rf_train = tc_ts_rf.head(-test_periods)
tc_ts_rf_test = tc_ts_rf.tail(test_periods)

# %%
train_data = TimeSeries.from_pd(tc_ts_pi_train)
test_data = TimeSeries.from_pd(tc_ts_pi_test)

# %%
from merlion.models.forecast.smoother import MSES, MSESConfig
from merlion.transform.resample import TemporalResample

model = MSES(
    MSESConfig(
        max_forecast_steps=100,
        max_backstep=60,
        transform=TemporalResample(granularity="4Q"),
    )
)
model.train(train_data=train_data)
test_pred, test_err = model.forecast(time_stamps=test_data.time_stamps)

# %%
from merlion.models.defaults import DefaultForecasterConfig, DefaultForecaster

model = MSES(DefaultForecasterConfig())
model.train(train_data=train_data)
test_pred, test_err = model.forecast(time_stamps=test_data.time_stamps)

# %%
tc_ts_pi_test

# %%
test_pred

# %%
test_err

# %%
import matplotlib.pyplot as plt

fig, ax = model.plot_forecast(time_series=test_data, plot_forecast_uncertainty=True)
plt.show()

# %%
# Evaluate the model's predictions quantitatively
from scipy.stats import norm
from merlion.evaluate.forecast import ForecastMetric

# Compute the sMAPE of the predictions (0 to 100, smaller is better)
smape = ForecastMetric.sMAPE.value(ground_truth=test_data, predict=test_pred)

# Compute the MSIS of the model's 95% confidence interval (0 to 100, smaller is better)
lb = TimeSeries.from_pd(test_pred.to_pd() + norm.ppf(0.025) * test_err.to_pd().values)
ub = TimeSeries.from_pd(test_pred.to_pd() + norm.ppf(0.975) * test_err.to_pd().values)
msis = ForecastMetric.MSIS.value(
    ground_truth=test_data, predict=test_pred, insample=train_data, lb=lb, ub=ub
)

rmse = ForecastMetric.RMSE.value(ground_truth=test_data, predict=test_pred)
print(f"sMAPE: {smape:.4f}, MSIS: {msis:.4f}, RMSE: {rmse:.4f}")

# %%
test_periods = 4
tc_ts_x = tc_ts[
    ["time_period", "research_funding_total", "investment_raised_total"]
].set_index("time_period")
train_data = TimeSeries.from_pd(tc_ts_x.head(-test_periods))
test_data = TimeSeries.from_pd(tc_ts_x.tail(test_periods))

# %%
from merlion.evaluate.forecast import ForecastMetric
from merlion.models.factory import ModelFactory
from merlion.models.ensemble.combine import ModelSelector

# Time series is sampled quarterly, so max_forecast_steps = 4 means we can predict up to 1 year in the future
target_seq_index = 1
max_forecast_steps = 4
kwargs = dict(target_seq_index=target_seq_index, max_forecast_steps=max_forecast_steps)

model = ModelFactory.create("DefaultForecaster", **kwargs)
train_pred, train_stderr = model.train(train_data)
target_univariate = test_data.univariates[test_data.names[target_seq_index]]
target = target_univariate[:max_forecast_steps].to_ts()
forecast, stderr = model.forecast(target.time_stamps)
rmse = ForecastMetric.RMSE.value(ground_truth=target, predict=forecast)
smape = ForecastMetric.sMAPE.value(ground_truth=target, predict=forecast)
print(f"{type(model).__name__}")
print(f"RMSE:  {rmse:.4f}")
print(f"sMAPE: {smape:.4f}")
print()

# %%
from merlion.evaluate.forecast import ForecastMetric
from merlion.models.factory import ModelFactory
from merlion.models.ensemble.combine import ModelSelector

# Time series is sampled quarterly, so max_forecast_steps = 4 means we can predict up to 1 year in the future
target_seq_index = 1
max_forecast_steps = 4
kwargs = dict(target_seq_index=target_seq_index, max_forecast_steps=max_forecast_steps)

model = ModelFactory.create("Arima", **kwargs)
train_pred, train_stderr = model.train(train_data)
target_univariate = test_data.univariates[test_data.names[target_seq_index]]
target = target_univariate[:max_forecast_steps].to_ts()
forecast, stderr = model.forecast(target.time_stamps)
rmse = ForecastMetric.RMSE.value(ground_truth=target, predict=forecast)
smape = ForecastMetric.sMAPE.value(ground_truth=target, predict=forecast)
print(f"{type(model).__name__}")
print(f"RMSE:  {rmse:.4f}")
print(f"sMAPE: {smape:.4f}")
print()

# %%
from merlion.models.forecast.prophet import Prophet

# %%
from merlion.evaluate.forecast import ForecastMetric
from merlion.models.factory import ModelFactory
from merlion.models.ensemble.combine import ModelSelector

# Time series is sampled quarterly, so max_forecast_steps = 4 means we can predict up to 1 year in the future
target_seq_index = 1
max_forecast_steps = 4
kwargs = dict(target_seq_index=target_seq_index, max_forecast_steps=max_forecast_steps)

model = ModelFactory.create("prophet", **kwargs)
train_pred, train_stderr = model.train(train_data)
target_univariate = test_data.univariates[test_data.names[target_seq_index]]
target = target_univariate[:max_forecast_steps].to_ts()
forecast, stderr = model.forecast(target.time_stamps)
rmse = ForecastMetric.RMSE.value(ground_truth=target, predict=forecast)
smape = ForecastMetric.sMAPE.value(ground_truth=target, predict=forecast)
print(f"{type(model).__name__}")
print(f"RMSE:  {rmse:.4f}")
print(f"sMAPE: {smape:.4f}")
print()

# %%

# %%
import matplotlib.pyplot as plt

fig, ax = model.plot_forecast(
    time_series=target_univariate, plot_forecast_uncertainty=True
)
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

for tech_cat in iss_ts.tech_category.unique():
    iss_ts_current_tc = iss_ts.query(f"tech_category == '{tech_cat}'")
    rf = iss_ts_current_tc.research_funding_total.values  # .diff().dropna()
    pi = iss_ts_current_tc.investment_raised_total.values
    stationarity_adf(rf, f"{tech_cat} research funding")
    try:
        stationarity_kpss(rf, f"{tech_cat} research funding")
    except OverflowError:
        print(f"{tech_cat} research funding divide by 0 error")
    stationarity_adf(pi, f"{tech_cat} investment")
    try:
        stationarity_kpss(pi, f"{tech_cat} investment")
    except OverflowError:
        print(f"{tech_cat} investment divide by 0 error")
    sm.graphics.tsa.plot_pacf(
        rf,
        lags=24,
        method="ywm",
        title=f"Partial Autocorrelation {tech_cat} research funding",
    )
    plt.show()
    sm.graphics.tsa.plot_pacf(
        pi,
        lags=24,
        method="ywm",
        title=f"Partial Autocorrelation {tech_cat} investment",
    )
    plt.show()

# %%
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import performance_metrics

# %%
tech_cat = "Carbon capture & storage"
tc_ts = iss_ts[iss_ts.tech_category == tech_cat]
test_periods = 4
df = tc_ts[["time_period", "investment_raised_total", "research_funding_total"]].rename(
    columns={
        "time_period": "ds",
        "investment_raised_total": "y",
        "research_funding_total": "rf",
    }
)
train = df.head(-test_periods)
test = df.tail(test_periods)
m = Prophet()
m.fit(df)
forecast = m.predict(pd.concat([train[["ds"]], test[["ds"]]]))
fig1 = m.plot(forecast)

# %%
train.tail(1)["ds"].values

# %%
df_fc = pd.merge(df, forecast[["ds", "yhat_lower", "yhat_upper", "yhat"]], on="ds")
df_fc["cutoff"] = train.tail(1)["ds"].values[0]
df_fc
performance_metrics(df_fc)

# %%
tech_cat = "Carbon capture & storage"
tc_ts = iss_ts[iss_ts.tech_category == tech_cat]
df = tc_ts[["time_period", "investment_raised_total", "research_funding_total"]].rename(
    columns={
        "time_period": "ds",
        "investment_raised_total": "y",
        "research_funding_total": "rf",
    }
)
m = Prophet()
m.add_regressor("rf")
m.fit(df)
forecast = m.predict(df[["ds", "rf"]])
fig2 = m.plot(forecast)

# %%
tech_cat = "Carbon capture & storage"
tc_ts = iss_ts[iss_ts.tech_category == tech_cat]
test_periods = 4
tc_ts_rf = tc_ts[["time_period", "research_funding_total"]].rename(
    columns={"time_period": "ds", "research_funding_total": "y"}
)
tc_ts_pi = tc_ts[["time_period", "investment_raised_total"]].rename(
    columns={"time_period": "ds", "investment_raised_total": "y"}
)
tc_ts_pi_train = tc_ts_pi.head(-test_periods)
tc_ts_pi_test = tc_ts_pi.tail(test_periods)
tc_ts_rf_train = tc_ts_rf.head(-test_periods)
tc_ts_rf_test = tc_ts_rf.tail(test_periods)

# %%
m = Prophet()
m.fit(tc_ts_rf_train)

# %%
forecast = m.predict(pd.concat([tc_ts_rf_train[["ds"]], tc_ts_rf_test[["ds"]]]))

# %%
fig1 = m.plot(forecast)

# %%
pd.merge(tc_ts_rf, forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds")


# %% [markdown]
# "Biomass heating" seems to work for research funding on its own...

# %%
tc_ts_rf

# %%
tc_ts_rf["rf"] = tc_ts_rf["y"].shift(1)
tc_ts_rf
