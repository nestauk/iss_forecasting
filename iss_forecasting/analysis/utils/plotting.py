"""Analysis utils for plotting
"""
import altair as alt
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
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
    data_source,
    x,
    y1,
    y2,
    chart_title,
    line_1_colour="#5276A7",
    line_2_colour="#57A44C",
):
    """Create an altair plot with two y axes and one x axis.

    Note that for x, y1 and y2 the data type can be specified
    as described here: https://altair-viz.github.io/user_guide/data.html

    Args:
        data_source (df): dataframe containing data to be plotted
        x (str): column containing data for x axis and data type
            e.g 'year:O'
        y1 (str): column containing data for y1 axis and data type
            e.g 'research_funding_total:Q'
        y2 (str): column containing data for y1 axis and data type
            e.g 'investment_raised_total:Q'
        line_1_colour (str): colour code for line 1, defaults to "#5276A7"
        line_2_colouur (str): colour code for line 2, defaults to "#57A44C"
        chart_title (str): title for chart

    Returns:
        altair.vegalite.v4.api.LayerChart: plot with two y axes and one x axis
    """
    base = alt.Chart(data_source, title=chart_title).encode(alt.X(x))
    line_1 = base.mark_line(color=line_1_colour).encode(
        alt.Y(y1, axis=alt.Axis(titleColor=line_1_colour))
    )
    line_2 = base.mark_line(color=line_2_colour).encode(
        alt.Y(y2, axis=alt.Axis(titleColor=line_2_colour))
    )
    return alt.layer(line_1, line_2).resolve_scale(y="independent")


def lagplot(x, y=None, lag=1, ax=None, **kwargs):
    """Plots a single lag plot for time series data.
    Can plot lagged x vs x or lagged x vs y.

    Args:
        x (pd.Series): x time series to be lagged
        y (pd.Series, optional): y time series, defaults to None. If None,
            y will be set to x
        lag (int, optional): how much to lag x time series by, defaults to 1.
        ax (int, optional): axes to draw the plot onto, defaults to None.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: Single lag plot

    This function is adapted from https://www.kaggle.com/ryanholbrook/time-series-as-features
    """
    x_ = x.shift(lag)
    y_ = y if y is not None else x
    corr = y_.corr(x_)
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
        **kwargs,
    )
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="medium"),
        frameon=True,
        loc="upper right",
    )
    ax.add_artist(at)
    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_size(8.5)
    return ax


def plot_lags(x, title, y=None, lags=6, nrows=1, lagplot_kwargs={}, **kwargs):
    """Plots multple lag plots for time series data.
    Can plot lagged x vs x or lagged x vs y.

    Args:
        x (pd.Series): x time series to be lagged
        title (str): title information to display in addition to
            number of lags
        y (pd.Series, optional): y time series, defaults to None.
            If None, y will be set to x
        lags (int, optional): how many lags to display, defaults to 6.
        nrows (int, optional): number of rows to split the plots over,
            defaults to 1.
        lagplot_kwargs (dict, optional): defaults to {}.

    Returns:
        matplotlib.figure.Figure: Multiple lag plots

    This function is adapted from https://www.kaggle.com/ryanholbrook/time-series-as-features
    """
    kwargs.setdefault("nrows", nrows)
    kwargs.setdefault("ncols", math.ceil((lags + 1) / nrows))
    kwargs.setdefault("figsize", (kwargs["ncols"] * 2, nrows * 2 + 0.5))
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k in zip(fig.get_axes(), range(kwargs["nrows"] * kwargs["ncols"])):
        if k <= lags:
            ax = lagplot(x, y, lag=k, ax=ax, **lagplot_kwargs)
            ax.set_title(f"Lag {k} {title}", fontdict=dict(fontsize=8.5))
            ax.set(xlabel="", ylabel="")
        else:
            ax.axis("off")
    plt.setp(axs[-1, :], xlabel=x.name)
    plt.setp(axs[:, 0], ylabel=y.name if y is not None else x.name)
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig
