"""Functions for creating causal graphs
"""
import pandas as pd


BEIS_CAUSES = "{n_grants,n_funding_rounds,n_months_before_first_investment,total_investment_amount_gbp,last_investment_round_gbp,has_received_grant,n_grants,has_received_ukri_grant,location_id,total_grant_amount_gbp,last_grant_amount_gbp,n_months_before_first_grant,future_success}"
INDUSTRY_CAUSES = "{n_months_before_first_investment,total_investment_amount_gbp,n_grants,has_received_grant,n_months_before_first_grant,has_received_ukri_grant,total_grant_amount_gbp,male_founder_percentage,last_grant_amount_gbp,future_success}"
INDUSTRY_CAUSED_BY = ["location_id", "founder_max_degrees", "founder_mean_degrees"]


def group_nodes_causes_nodes(caused_by_nodes: list, causes_nodes: str) -> str:
    """Generate causal graph in a format that can be used by DoWhy.

    Args:
        caused_by_nodes: List of nodes that cause the 'causes_nodes'
        causes_nodes: String in the format '{node1,node2,node3,...,nodex}'
            of nodes that are caused by the 'caused_by_nodes'

    Returns:
        Causal graph string

    Example:
        caused_by_nodes = ['node1', 'node2']
        causes_nodes = '{node3,node4}'
        group_nodes_causes_nodes(caused_by_nodes, causes_nodes) ->
        'node1 -> {node3,node4};node2 -> {node3,node4};'
    """
    string = ""
    for group_node in caused_by_nodes:
        string = f"{string}{group_node} -> {causes_nodes};"
    return string


def beis_out_graph(data: pd.DataFrame, beis_causes: str = BEIS_CAUSES) -> str:
    """Return causal graph string for the edges coming out of all the beis nodes"""
    beis_cols = [col for col in data.columns if "beis_" in col]
    return group_nodes_causes_nodes(beis_cols, beis_causes)


def industry_cols(data: pd.DataFrame) -> list:
    """Return list of columns containing group_"""
    return [col for col in data.columns if "group_" in col]


def industry_out_graph(
    data: pd.DataFrame, industry_causes: str = INDUSTRY_CAUSES
) -> str:
    """Return causal graph string for the edges coming out of the industry nodes"""
    ind_cols = industry_cols(data)
    return group_nodes_causes_nodes(ind_cols, industry_causes)


def industry_in_graph(
    data: pd.DataFrame, industry_caused_by: str = INDUSTRY_CAUSED_BY
) -> str:
    """Return causal graph string for the edges coming into the industry nodes"""
    industry_cols_string = "{"
    for ind_col in industry_cols(data):
        industry_cols_string = industry_cols_string + ind_col + ","
    industry_cols_string = industry_cols_string[:-1] + "}"
    return group_nodes_causes_nodes(industry_caused_by, industry_cols_string)


def small_graph() -> str:
    """Return small causal graph string containing only a few nodes"""
    return """digraph {
future_success[label="Future Success"];
founder_mean_degrees[label="Founders Mean Degrees"];
total_investment_amount_gbp[label="Total investment amount"];
has_received_ukri_grant[label="Has received a UKRI grant"];
has_received_grant[label="Has received a grant"];
has_received_ukri_grant -> {total_investment_amount_gbp,future_success};
has_received_grant -> {total_investment_amount_gbp,future_success};
total_investment_amount_gbp -> future_success;
founder_mean_degrees -> {has_received_grant,has_received_ukri_grant,future_success};
}"""


def medium_graph() -> str:
    """Return medium causal graph string containing all nodes except for
    the beis and industry related nodes. This graph is useful for
    visualising as the large graph cannot be visualised."""
    return """digraph {
future_success[label="Future Success"];
founder_max_degrees[label="Founders Max Degrees"];
founder_mean_degrees[label="Founders Mean Degrees"];
has_received_ukri_grant[label="Has received a UKRI grant"];
has_received_grant[label="Has received a grant"];
n_grants[label="Number of grants received"];
n_months_before_first_grant[label="Months before first grant"];
n_months_before_first_investment[label="Months before first investment"];
n_months_since_last_investment[label="Months since last investment"]
n_months_since_last_grant[label="Months since last grant"];
last_grant_amount_gbp[label="Last grant amount"];
total_grant_amount_gbp[label="Total grant amount"];
total_investment_amount_gbp[label="Total investment amount"];
last_investment_round_gbp[label="Last investment round amount"];
n_unique_investors_total[label="Number investors total"];
n_unique_investors_last_round[label="Number of investors last round"];
male_founder_percentage[label="Male founder %"];
n_funding_rounds[label="Investment rounds"];
founder_count[label="Number of founders"];
last_investment_round_type[label="Last investment round type"];
n_months_since_founded[label="Months Since Founded"];
location_id[label="Company location"];
founder_max_degrees -> {n_grants,has_received_grant,has_received_ukri_grant,n_months_before_first_grant,n_months_before_first_investment,future_success};
founder_mean_degrees -> {n_grants,has_received_grant,has_received_ukri_grant,n_months_before_first_grant,n_months_before_first_investment,future_success};
n_months_before_first_grant -> n_months_before_first_investment;
n_months_since_last_grant -> future_success;
has_received_ukri_grant -> {total_investment_amount_gbp,n_grants,future_success};
last_grant_amount_gbp -> {total_grant_amount_gbp,future_success};
total_grant_amount_gbp -> future_success;
n_grants -> {total_investment_amount_gbp,future_success};
has_received_grant -> {total_investment_amount_gbp,future_success,n_grants};
last_investment_round_gbp -> {total_investment_amount_gbp,future_success};
n_unique_investors_total -> {total_investment_amount_gbp,future_success};
n_unique_investors_last_round -> {n_unique_investors_total,future_success};
n_months_since_last_investment -> future_success;
total_investment_amount_gbp -> future_success;
male_founder_percentage -> future_success;
n_funding_rounds -> {total_investment_amount_gbp,n_unique_investors_total,future_success};
founder_count -> n_months_before_first_investment;
n_months_before_first_investment -> future_success;
last_investment_round_type -> {last_investment_round_gbp,future_success};
n_months_since_founded -> {total_grant_amount_gbp,last_investment_round_type,total_investment_amount_gbp,n_funding_rounds,future_success};
location_id -> {has_received_grant,n_funding_rounds,total_investment_amount_gbp,n_months_before_first_investment,future_success};
}"""


def large_graph(data: pd.DataFrame) -> str:
    """Return the large causal graph by taking the medium graph
    and adding causal relationships for the beis nodes and the
    industry nodes"""
    med_graph_trim = medium_graph()[:-1]
    add_ind_beis = """
%s
%s
%s
}""" % (
        industry_out_graph(data),
        beis_out_graph(data),
        industry_in_graph(data),
    )
    return med_graph_trim + add_ind_beis
