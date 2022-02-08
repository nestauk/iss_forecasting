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
# # Using generated data with time series process in research_funding_and_investment.ipynb
# This notebook will explore the differences between when time series data is grouped by months, quarters and years.

# %%
from iss_forecasting.getters.iss_green_pilot_ts import get_iss_green_pilot_time_series
from iss_forecasting.utils.processing import find_zero_items
from iss_forecasting.analysis.utils.plotting import plot_two_y_one_x, lagplot, plot_lags
from iss_forecasting.utils.stats_tests import stationarity_adf, stationarity_kpss
from iss_forecasting.analysis.utils.time_series import generate_ts, groupby_time_period
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
# Generate monthly data
gen_ts_months = generate_ts(period="M", lag=12)

# %%
# Group monthly data into quarterly and yearly data
gen_ts_quarters = groupby_time_period(gen_ts_months, "Q")
gen_ts_years = groupby_time_period(gen_ts_months, "Y")
tss = [gen_ts_months, gen_ts_quarters, gen_ts_years]

# %%
# Plot monthly, quarterly, yearly generated time series
plots_ts = alt.vconcat()
for ts in tss:
    plots_ts &= plot_two_y_one_x(
        data_source=ts,
        x="time_period:T",
        y1="research_funding_total:Q",
        y2="investment_raised_total:Q",
        chart_title=f"{ts.tech_category.unique()[0]} research funding vs. investment over time",
    )
plots_ts

# %% [markdown]
# # Plot cross correlations for monthly, quarterly and yearly generated data
# Here, we would expect the highest cross correlation values at:
# * lag 12 for monthly
# * lag 4 for quarterly
# * lag 1 for yearly

# %%
lagss = [100, 40, 9]
nrowss = [14, 6, 2]
for ts, lags, nrows in zip(tss, lagss, nrowss):
    plot = plot_lags(
        x=ts.research_funding_total,
        title=ts.tech_category.unique()[0],
        y=ts.investment_raised_total,
        lags=lags,
        nrows=nrows,
    )

# %% [markdown]
# The highest correlations for monthly data is at a lag of 11-13 months<br>
# The highest correlations for the quarterly data is at a lag of 4 quarters.<br>
# The highest correlations for yearly data is at lag of 1 year. <br>
# The confidence intervals tend to be larger in the yearly data compared to the quarterly data which tend to be larger than the mothly data.

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

# %%
# Create 1d yearly time series
rf_yearly = gen_ts_years.research_funding_total
pi_yearly = gen_ts_years.investment_raised_total
# Create 1d quarterly time series
rf_quarterly = gen_ts_quarters.research_funding_total
pi_quarterly = gen_ts_quarters.investment_raised_total
# Create 1d monthly time series
rf_monthly = gen_ts_months.research_funding_total
pi_monthly = gen_ts_months.investment_raised_total

# %%
# Test stationarity yearly
title = "Yearly"
stationarity_adf(rf_yearly, f"{title} research funding")
stationarity_kpss(rf_yearly, f"{title} research funding")
stationarity_adf(pi_yearly, f"{title} investment")
stationarity_kpss(pi_yearly, f"{title} investment")

# %%
# Diff yearly research funding and retest for stationarity
rf_yearly_diff = rf_yearly.diff().dropna()
title = "Yearly (diff)"
stationarity_adf(rf_yearly_diff, f"{title} research funding")
stationarity_kpss(rf_yearly_diff, f"{title} research funding")

# %%
# Test stationarity quarterly
title = "Quarterly"
stationarity_adf(rf_quarterly, f"{title} research funding")
stationarity_kpss(rf_quarterly, f"{title} research funding")
stationarity_adf(pi_quarterly, f"{title} investment")
stationarity_kpss(pi_quarterly, f"{title} investment")

# %%
# Test stationarity monthly
title = "Monthly"
stationarity_adf(rf_monthly, f"{title} research funding")
stationarity_kpss(rf_monthly, f"{title} research funding")
stationarity_adf(pi_monthly, f"{title} investment")
stationarity_kpss(pi_monthly, f"{title} investment")

# %%
# Combine research funding and private investment time series yearly, drop last row due to diff
combine_yearly = pd.concat([rf_yearly_diff, pi_yearly], axis=1).tail(-1)
# Check for granger causality research funding -> private investment yearly
rf_predict_pi_yearly = grangercausalitytests(
    combine_yearly[["investment_raised_total", "research_funding_total"]], maxlag=4
)

# %% [markdown]
# The yearly tests suggest that research funding granger causes investment at lags of 1, 2 and 3 years. Would have expected research funding granger causing investment at a lag of 1 only?

# %%
# Check for granger causality private investment -> research funding yearly
pi_predict_rf_yearly = grangercausalitytests(
    combine_yearly[["research_funding_total", "investment_raised_total"]], maxlag=4
)

# %% [markdown]
# The yearly tests suggest that investment does not granger cause research funding at any lags, which is what we would expect.

# %%
# Combine research funding and private investment time series quarterly
combine_quarterly = pd.concat([rf_quarterly, pi_quarterly], axis=1)
# Check for granger causality research funding -> private investment quarterly
rf_predict_pi_quarterly = grangercausalitytests(
    combine_quarterly[["investment_raised_total", "research_funding_total"]], maxlag=18
)

# %% [markdown]
# The quarterly tests suggest that research funding granger causes investment at lag 4 as expected but also finds the same for lags 5-17.

# %%
# Check for granger causality private investment -> research funding quarterly
pi_predict_rf_quarterly = grangercausalitytests(
    combine_quarterly[["research_funding_total", "investment_raised_total"]], maxlag=18
)

# %% [markdown]
# Unexpectedly, the quarterly tests also suggest that investment granger causes research funding from lags 4-14.

# %%
# Combine research funding and private investment time series monthly
combine_monthly = pd.concat([rf_monthly, pi_monthly], axis=1)
# Check for granger causality research funding -> private investment monthly
rf_predict_pi_monthly = grangercausalitytests(
    combine_monthly[["investment_raised_total", "research_funding_total"]], maxlag=55
)

# %% [markdown]
# The monthly tests suggests research funding granger causes investment from the 11th-50th lag. Again, we would expect the tests to have low p values for lag 12, but then it also it also has low p values up to lag 50?

# %%
# Check for granger causality private investment -> research funding monthly
pi_predict_rf_monthly = grangercausalitytests(
    combine_monthly[["research_funding_total", "investment_raised_total"]], maxlag=55
)

# %% [markdown]
# The monthly tests also suggests investment granger causes research funding from the 11th-33rd lag. Also unexpected?

# %%
