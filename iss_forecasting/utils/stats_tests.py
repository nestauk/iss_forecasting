"""Statistical testing utils"""
from statsmodels.tsa.stattools import adfuller, kpss
from numpy.typing import ArrayLike


def stationarity_adf(time_series: ArrayLike, name: str) -> None:
    """Check if a time series is stationary based on ADF test
    and print the results

    Args:
        time_series: Time series to be checked (must be 1 dimensional)
        name: Name of time series
    """
    results = adfuller(time_series)
    p_value = round(results[1], 3)
    n_lags = results[2]
    print(
        f"{name} ADF test: p_value is {p_value}. {p_value}"
        f"{' < 0.05, time series is' if p_value < 0.05 else ' > 0.05, time series is not'}"
        f" stationary. {n_lags} lags"
    )


def stationarity_kpss(time_series: ArrayLike, name: str) -> None:
    """Check if a time series is stationary based on KPSS test
    and print the results

    Args:
        time_series: Time series to be checked (must be 1 dimensional)
        name: Name of time series
    """
    results = kpss(time_series)
    p_value = round(results[1], 3)
    n_lags = results[2]
    print(
        f"{name} KPSS test: p_value is {p_value}. {p_value}"
        f"{' < 0.05, time series is not' if p_value < 0.05 else ' > 0.05, time series is'}"
        f" stationary. {n_lags} lags"
    )
