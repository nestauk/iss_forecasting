"""Analysis utils for plotting
"""
import altair as alt


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
