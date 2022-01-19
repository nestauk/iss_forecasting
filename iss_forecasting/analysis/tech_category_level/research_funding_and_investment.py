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
# ## Exploring the relationship between research funding and private investment
# This notebook looks at the relationship between research funding and private investment. It uses pilot data produced from [Innovation Sweet Spots](https://github.com/nestauk/innovation_sweet_spots) that focuses on companies in technology sectors that could help tackle climate change. The underlying research funding data comes from UKRI's Gateway to Research and private investment data comes from Crunchbase.
# %%
from iss_forecasting.getters.iss_green_pilot_ts import get_iss_green_pilot_time_series
from iss_forecasting.utils.processing import find_zero_items
from iss_forecasting.analysis.utils.plotting import plot_two_y_one_x, lagplot, plot_lags
from iss_forecasting.utils.stats_tests import stationarity_adf, stationarity_kpss
import altair as alt
import pandas as pd
from statsmodels.tsa.stattools import ccf

# %%
# load iss green time series data
iss_ts = get_iss_green_pilot_time_series()

# %%
# view sample of dataframe
iss_ts.tail(5)

# %% [markdown]
# Column descriptions:<br>
# `year` = year (between 2007 and 2021)<br>
# `research_no_of_projects` = number of research projects started in a specific `year` (GtR)<br>
# `research_funding_total` = research funding amount (GBP 1000s) awarded in a specific `year` (GtR)<br>
# `investment_rounds` = number of rounds of investment in a specific `year` (Crunchbase)<br>
# `investment_raised_total` = raised investment amount (GBP 1000s) in a specific `year` (Crunchbase)<br>
# `no_of_orgs_founded` = number of new companies started in the `year` (Crunchbase)<br>
# `articles` = number of articles in the Guardian with the technology keywords and phrases<br>
# `speeches` = number of speeches in Hansard with the technology keywords and phrases<br>
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
# %%
# plot research funding vs private investment for each tech category
tech_cats = iss_ts.tech_category.unique()
plots_ts = alt.vconcat()
for tech_cat in tech_cats:
    iss_ts_current_tc = iss_ts.query(f"tech_category == '{tech_cat}'")
    plots_ts &= plot_two_y_one_x(
        data_source=iss_ts_current_tc,
        x="year:O",
        y1="research_funding_total:Q",
        y2="investment_raised_total:Q",
        chart_title=f"{tech_cat} research funding vs. investment over time",
    )

plots_ts

# %% [markdown]
# Looking at the above plots, there does not seem to be an obvious relationship between research funding and private investment that is consistent across all tech categories.

# %%
# plot research funding lagged against private investment for each technology category
for tech_cat in iss_ts.tech_category.unique():
    iss_ts_current_tc = iss_ts.query(f"tech_category == '{tech_cat}'")
    plot = plot_lags(
        x=iss_ts_current_tc.research_funding_total,
        title=tech_cat,
        y=iss_ts_current_tc.investment_raised_total,
        lags=8,
        nrows=1,
    )

# %% [markdown]
# Looking at the above lag plots, there does not seem to be a common time lag between research funding and private investment.<br>
# The Batteries technology category has the highest correlation values with high correlation between private investment and research funding lagged by 4 years+.

# %%
for tech_cat in iss_ts.tech_category.unique():
    tech_cat_df = iss_ts.query(f"tech_category == '{tech_cat}'")
    stationarity_adf(tech_cat_df.research_funding_total, tech_cat + " research funding")
    stationarity_kpss(
        tech_cat_df.research_funding_total, tech_cat + " research funding"
    )
    stationarity_adf(tech_cat_df.investment_raised_total, tech_cat + " investment")
    stationarity_kpss(tech_cat_df.investment_raised_total, tech_cat + " investment")
    print("*" * 10)

# %%

# %%

# %%
