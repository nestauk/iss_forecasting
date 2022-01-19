"""Statistical testing utils
"""
from statsmodels.tsa.stattools import adfuller, kpss


def stationarity_adf(time_series, name):
    "Check if the time series is stationary based on ADF test"
    results = adfuller(time_series)
    p_value = round(results[1], 3)
    n_lags = results[2]
    print(
        f"{name} ADF test: p_value is {p_value}. {p_value}"
        f'{" < 0.05, time series is" if p_value < 0.05 else " > 0.05, time series is not"}'
        f" stationary. {n_lags} lags"
    )


def stationarity_kpss(time_series, name):
    "Check if the time series is stationary based on KPSS test"
    results = kpss(time_series)
    p_value = round(results[1], 3)
    n_lags = results[2]
    print(
        f"{name} KPSS test: p_value is {p_value}. {p_value}"
        f'{" < 0.05, time series is not" if p_value < 0.05 else " > 0.05, time series is"}'
        f" stationary. {n_lags} lags"
    )
