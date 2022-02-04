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
"""Move these function to python code, trim down generate_ts?"""


def generate_ts(
    period: str, lag: int, start: str = "01/01/2007", end: str = "12/31/2021"
) -> pd.DataFrame:
    np.random.seed(40)
    dates = pd.period_range(start, end, freq=period).to_timestamp()
    tech_category = f"Generated {period}"
    research_funding_total = [abs(np.random.normal()) * 100]
    n_dates = len(dates)
    for _ in range(n_dates):
        research_funding_total.append(
            research_funding_total[-1] * 1.1 * abs(np.random.normal())
            + 200 * abs(np.random.normal())
        )
    investment_raised_total = [
        item * 2 * abs(np.random.normal()) for item in research_funding_total
    ]
    research_funding_total = research_funding_total[lag:]
    investment_raised_total = investment_raised_total[:-lag]
    dates = dates[lag - 1 :]
    n_dates = len(dates)
    gen_ts_df = pd.DataFrame(
        {
            "time_period": dates,
            "research_funding_total": research_funding_total,
            "investment_raised_total": investment_raised_total,
            "tech_category": [tech_category] * n_dates,
        }
    )
    gen_ts_df["time_period"] = gen_ts_df["time_period"].dt.strftime("%Y-%m-%d")
    return gen_ts_df.astype({"time_period": "datetime64[ns]"})


def groupby_time_period(time_series: pd.DataFrame, period: str) -> pd.DataFrame:
    grouped = time_series.groupby(time_series["time_period"].dt.to_period(period)).agg(
        "sum"
    )
    grouped.index = grouped.index.astype("datetime64[ns]")
    grouped["tech_category"] = f"Generated {period}"
    return grouped.reset_index()


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
# """Add a comment on how the confidence intervals change across years / months / quarters"""

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

# %%
# Check for granger causality private investment -> research funding yearly
pi_predict_rf_yearly = grangercausalitytests(
    combine_yearly[["research_funding_total", "investment_raised_total"]], maxlag=4
)

# %%
# Combine research funding and private investment time series quarterly
combine_quarterly = pd.concat([rf_quarterly, pi_quarterly], axis=1)
# Check for granger causality research funding -> private investment quarterly
rf_predict_pi_quarterly = grangercausalitytests(
    combine_quarterly[["investment_raised_total", "research_funding_total"]], maxlag=18
)

# %%
# Check for granger causality private investment -> research funding quarterly
pi_predict_rf_quarterly = grangercausalitytests(
    combine_quarterly[["research_funding_total", "investment_raised_total"]], maxlag=18
)

# %%
# Combine research funding and private investment time series monthly
combine_monthly = pd.concat([rf_monthly, pi_monthly], axis=1)
# Check for granger causality research funding -> private investment monthly
rf_predict_pi_monthly = grangercausalitytests(
    combine_monthly[["investment_raised_total", "research_funding_total"]], maxlag=55
)

# %%
# Check for granger causality private investment -> research funding monthly
pi_predict_rf_monthly = grangercausalitytests(
    combine_monthly[["research_funding_total", "investment_raised_total"]], maxlag=55
)

# %% [markdown]
# The granger causality tests for the monthly data finds:
# * research funding granger causes investment from the 11th lag+
# * also finds investment granger causes research funding from the 11th lag+.

# %%
