"""Analysis utils for generating and manipulating time series data
"""
import numpy as np
import pandas as pd


def generate_ts(
    period: str,
    lag: int,
    start: str = "01/01/2007",
    end: str = "12/31/2021",
    rf_scale: float = 100,
    rf_const: float = 200,
    factor: float = 1.1,
    pi_scale: float = 2,
) -> pd.DataFrame:
    """Generate dataframe containing columns for time_period, research_funding_total,
    investment_raised_total and tech_category. research_funding_total and
    investment_raised_total have generated time series data, where investment_raised_total
    is research_funding_total shifted by the lag variable.

    Args:
        period: Time period to group the data by, 'M', 'Q' or 'Y'
        lag: Number of time periods to have research_funding_total lead to investment_raised_total
        start: Start date. Defaults to "01/01/2007".
        end: End date. Defaults to "12/31/2021".

    Returns:
        Generated time series data for investment_raised_total and research_funding_total.
    """
    np.random.seed(40)
    dates = pd.period_range(start, end, freq=period).to_timestamp()
    tech_category = f"Generated {period}"
    research_funding_total = [abs(np.random.normal()) * rf_scale]
    n_dates = len(dates)
    for _ in range(n_dates):
        research_funding_total.append(
            research_funding_total[-1] * factor * abs(np.random.normal())
            + rf_const * abs(np.random.normal())
        )
    investment_raised_total = [
        item * pi_scale * abs(np.random.normal()) for item in research_funding_total
    ]
    research_funding_total = research_funding_total[lag:]
    investment_raised_total = investment_raised_total[:-lag]
    dates = dates[lag - 1 :]
    n_dates = len(dates)
    return (
        pd.DataFrame(
            {
                "time_period": dates,
                "research_funding_total": research_funding_total,
                "investment_raised_total": investment_raised_total,
                "tech_category": [tech_category] * n_dates,
            }
        )
        .assign(time_period=lambda x: x.time_period.dt.strftime("%Y-%m-%d"))
        .astype({"time_period": "datetime64[ns]"})
    )


def groupby_time_period(time_series_df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Returns dataframe grouped by time period

    Args:
        time_series: Dataframe containing column for time_period
        period: Time period to group the data by, 'M', 'Q' or 'Y'

    Returns:
        dataframe grouped by time period
    """
    grouped = time_series_df.groupby(
        time_series_df["time_period"].dt.to_period(period)
    ).agg("sum")
    grouped.index = grouped.index.astype("datetime64[ns]")
    grouped["tech_category"] = f"Generated {period}"
    return grouped.reset_index()
