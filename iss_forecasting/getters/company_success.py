"""Data getters for company future success prediction"""
import pandas as pd
from iss_forecasting import PROJECT_DIR


def get_company_future_success_dataset(
    date_range: str = "2011-01-01-2019-01-01",
    uk_only: bool = True,
    test: bool = False,
    split: int = None,
) -> pd.DataFrame:
    """Load company level future success dataset

    Args:
        date_range: Date range in the format - 2011-01-01-2019-01-01
        uk_only: True to load a uk only data, False to load a worldwide data
        test: True to load a test dataset, False to load a full dataset
        split: Int relating to the split value of the file to be loaded

    Returns:
        Dataset of companies including information about investments,
        grants and future success
    """
    test_indicator = "_test" if test else ""
    region_indicator = "_ukonly" if uk_only else "_worldwide"
    split_indicator = f"_split_{split}" if split else ""
    return pd.read_csv(
        PROJECT_DIR
        / f"inputs/data/company_level/company_data_window_{date_range}{test_indicator}{region_indicator}{split_indicator}.csv",
        index_col=0,
        dtype={
            "location_id": "category",
            "has_email": "uint8",
            "has_phone": "uint8",
            "has_facebook_url": "uint8",
            "has_twitter_url": "uint8",
            "has_homepage_url": "uint8",
            "has_linkedin_url": "uint8",
            "group_administrative_services": "uint8",
            "group_advertising": "uint8",
            "group_agriculture_and_farming": "uint8",
            "group_apps": "uint8",
            "group_artificial_intelligence": "uint8",
            "group_biotechnology": "uint8",
            "group_clothing_and_apparel": "uint8",
            "group_commerce_and_shopping": "uint8",
            "group_community_and_lifestyle": "uint8",
            "group_consumer_electronics": "uint8",
            "group_consumer_goods": "uint8",
            "group_content_and_publishing": "uint8",
            "group_data_and_analytics": "uint8",
            "group_design": "uint8",
            "group_education": "uint8",
            "group_energy": "uint8",
            "group_events": "uint8",
            "group_financial_services": "uint8",
            "group_food_and_beverage": "uint8",
            "group_gaming": "uint8",
            "group_government_and_military": "uint8",
            "group_hardware": "uint8",
            "group_health_care": "uint8",
            "group_information_technology": "uint8",
            "group_internet_services": "uint8",
            "group_lending_and_investments": "uint8",
            "group_manufacturing": "uint8",
            "group_media_and_entertainment": "uint8",
            "group_messaging_and_telecommunications": "uint8",
            "group_mobile": "uint8",
            "group_music_and_audio": "uint8",
            "group_natural_resources": "uint8",
            "group_navigation_and_mapping": "uint8",
            "group_no_industry_listed": "uint8",
            "group_other": "uint8",
            "group_payments": "uint8",
            "group_platforms": "uint8",
            "group_privacy_and_security": "uint8",
            "group_professional_services": "uint8",
            "group_real_estate": "uint8",
            "group_sales_and_marketing": "uint8",
            "group_science_and_engineering": "uint8",
            "group_software": "uint8",
            "group_sports": "uint8",
            "group_sustainability": "uint8",
            "group_transportation": "uint8",
            "group_travel_and_tourism": "uint8",
            "group_video": "uint8",
            "future_success": "uint8",
            "n_funding_rounds": "uint16",
            "last_investment_round_type": "category",
            "last_investment_round_gbp": "float32",
            "n_months_before_first_investment": "uint8",
            "total_investment_amount_gbp": "float64",
            "n_months_since_last_investment": "uint8",
            "n_months_since_founded": "uint8",
            "n_unique_investors_last_round": "float16",
            "n_unique_investors_total": "float16",
            "founder_count": "float16",
            "male_founder_percentage": "float16",
            "founder_max_degrees": "float16",
            "founder_mean_degrees": "float16",
        },
    )
