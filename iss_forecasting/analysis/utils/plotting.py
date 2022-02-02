"""Analysis utils for plotting
"""
from typing import Optional
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from altair.vegalite.v4.api import LayerChart
import seaborn as sns
import pingouin as pg
import math

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
)


def plot_two_y_one_x(
    data_source: pd.DataFrame,
    x: str,
    y1: str,
    y2: str,
    chart_title: str,
    line_1_colour: str = "#5276A7",
    line_2_colour: str = "#57A44C",
) -> LayerChart:
    """Create an altair plot with two y axes and one x axis.

    Note that for x, y1 and y2 the data type can be specified
    as described here: https://altair-viz.github.io/user_guide/data.html

    Args:
        data_source: Dataframe containing data to be plotted
        x: Column containing data for x axis and data type
            e.g 'year:O'
        y1: Column containing data for y1 axis and data type
            e.g 'research_funding_total:Q'
        y2: Column containing data for y1 axis and data type
            e.g 'investment_raised_total:Q'
        line_1_colour: Colour code for line 1, defaults to "#5276A7"
        line_2_colour: Colour code for line 2, defaults to "#57A44C"
        chart_title: Title for chart

    Returns:
        Plot with two y axes and one x axis
    """
    base = alt.Chart(data_source, title=chart_title).encode(alt.X(x))
    line_1 = base.mark_line(color=line_1_colour).encode(
        alt.Y(y1, axis=alt.Axis(titleColor=line_1_colour))
    )
    line_2 = base.mark_line(color=line_2_colour).encode(
        alt.Y(y2, axis=alt.Axis(titleColor=line_2_colour))
    )
    return alt.layer(line_1, line_2).resolve_scale(y="independent")


def lagplot(
    x: pd.Series,
    y: Optional[pd.Series] = None,
    lag: int = 1,
    ax: Optional[int] = None,
) -> plt.Axes:
    """Plots a single lag plot for time series data.
    Can plot lagged x vs x or lagged x vs y.

    Args:
        x: X axis time series to be lagged
        y: Y axis time series, defaults to None. If None, y will be set to x.
        lag: How much to lag x time series by, defaults to 1.
        ax: Axes to draw the plot onto, defaults to None.

    Returns:
        Single lag plot

    This function is adapted from https://www.kaggle.com/ryanholbrook/time-series-as-features
    """
    x_ = x.shift(lag)
    y_ = y if y is not None else x
    pg_corr = pg.corr(x_, y_).round(2)
    r = pg_corr["r"].values[0]
    ci = pg_corr["CI95%"].values[0]
    if ax is None:
        _, ax = plt.subplots()
    scatter_kws = dict(
        alpha=0.75,
        s=3,
    )
    line_kws = dict(
        color="C3",
    )
    ax = sns.regplot(
        x=x_,
        y=y_,
        scatter_kws=scatter_kws,
        line_kws=line_kws,
        lowess=False,
        order=1,
        ci=None,
        ax=ax,
    )
    at = AnchoredText(
        f"r:{r}, ci:{ci}",
        prop=dict(size=8.5, fontweight="bold"),
        frameon=True,
        loc="upper right",
    )
    ax.add_artist(at)
    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_size(8.5)
    return ax


def plot_lags(
    x: pd.Series,
    title: str,
    y: Optional[pd.Series] = None,
    lags: int = 6,
    nrows: int = 1,
) -> plt.Figure:
    """Plots multple lag plots for time series data.
    Can plot lagged x vs x or lagged x vs y.

    Args:
        x: X axis time series to be lagged
        title: Title information to display in addition to number of lags
        y: Y axis time series, defaults to None. If None, y will be set to x
        lags: How many lags to display, defaults to 6.
        nrows: Number of rows to split the plots over, defaults to 1.

    Returns:
        Multiple lag plots

    This function is adapted from https://www.kaggle.com/ryanholbrook/time-series-as-features
    """
    ncols = math.ceil((lags + 1) / nrows)
    fig, axs = plt.subplots(
        sharex=True,
        sharey=True,
        squeeze=False,
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 2, nrows * 2 + 0.5),
    )
    for ax, k in zip(fig.get_axes(), range(nrows * ncols)):
        if k <= lags:
            ax = lagplot(x, y, lag=k, ax=ax)
            ax.set_title(f"Lag {k} {title}", fontdict=dict(fontsize=8.5))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis("off")
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig
