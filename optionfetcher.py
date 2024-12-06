import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, TypedDict, List, Dict

import pandas as pd
import requests
from fake_useragent import UserAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Data Definitions

class OptionData(TypedDict):
    insCode_P: str
    insCode_C: str
    contractSize: int
    uaInsCode: str
    lVal18AFC_P: str
    lVal30_P: str
    zTotTran_P: int
    qTotTran5J_P: int
    qTotCap_P: float
    notionalValue_P: float
    pClosing_P: int
    priceYesterday_P: int
    oP_P: int
    pDrCotVal_P: int
    lval30_UA: str
    pClosing_UA: int
    priceYesterday_UA: int
    beginDate: str
    endDate: str
    strikePrice: int
    remainedDay: int
    pDrCotVal_C: int
    oP_C: int
    pClosing_C: int
    priceYesterday_C: int
    notionalValue_C: float
    qTotCap_C: float
    qTotTran5J_C: int
    zTotTran_C: int
    lVal30_C: str
    lVal18AFC_C: str
    pMeDem_P: int
    qTitMeDem_P: int
    pMeOf_P: int
    qTitMeOf_P: int
    pMeDem_C: int
    qTitMeDem_C: int
    pMeOf_C: int
    qTitMeOf_C: int
    yesterdayOP_C: int
    yesterdayOP_P: int


class OptionDataOutput(TypedDict):
    instrumentOptMarketWatch: List[OptionData]


# Constants

MARKETS_NUM: Dict[str, int] = {
    "bourse": 1,
    "fara_bourse": 2
}

GENERAL_COLUMN_NAMES: Dict[str, str] = {
    "uaInsCode": "ua_tse_code",
    "lval30_UA": "ua_ticker",
    "remainedDay": "days_to_maturity",
    "strikePrice": "strike_price",
    "contractSize": "contract_size",
    "pClosing_UA": "ua_close_price",
    "priceYesterday_UA": "ua_yesterday_price",
    "beginDate": "begin_date",
    "endDate": "end_date",
}

SPECIFIC_COLUMN_NAMES: Dict[str, str] = {
    'insCode': "tse_code",
    'lVal18AFC': "ticker",
    'zTotTran': "trades_num",
    'qTotTran5J': "trades_volume",
    'qTotCap': "trades_value",
    'pDrCotVal': "last_price",
    'pClosing': "close_price",
    'priceYesterday': "yesterday_price",
    'oP': "open_positions",
    'yesterdayOP': "yesterday_open_positions",
    'notionalValue': "notional_value",
    'pMeDem': "bid_price",
    'qTitMeDem': "bid_volume",
    'pMeOf': "ask_price",
    'qTitMeOf': "ask_volume",
    'lVal30': "name",
}

# User-Agent Setup

fake_user_agent = UserAgent()

# Data Fetching

def fetch_option_data(market: Literal["bourse", "fara_bourse"]) -> List[OptionData]:
    """
    Fetch option data for a given market.

    Args:
        market (Literal["bourse", "fara_bourse"]): The market to fetch data from.

    Returns:
        List[OptionData]: A list of option data dictionaries.
    """
    logger.info(f"Fetching option data for market: {market}")
    market_num = MARKETS_NUM.get(market)

    if market_num is None:
        logger.error(f"Invalid market value: {market}. Expected 'bourse' or 'fara_bourse'.")
        raise ValueError("Invalid market value. Expected 'bourse' or 'fara_bourse'.")

    url = f"https://cdn.tsetmc.com/api/Instrument/GetInstrumentOptionMarketWatch/{market_num}"
    headers = {'User-Agent': fake_user_agent.random}
    
    try:
        response = requests.get(url=url, headers=headers, timeout=10)
        response.raise_for_status()
        json_response: OptionDataOutput = response.json()
        data = json_response.get("instrumentOptMarketWatch", [])
        logger.info(f"Fetched {len(data)} records for market: {market}")
        return data
    except requests.RequestException as e:
        logger.error(f"An error occurred while fetching option data for market {market}: {e}")
        raise

def fetch_entire_market_data() -> List[OptionData]:
    """
    Fetch option data for all specified markets concurrently.

    Returns:
        List[OptionData]: Combined list of option data from all markets.
    """
    logger.info("Starting to fetch entire market data concurrently.")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(fetch_option_data, market): market
            for market in MARKETS_NUM.keys()
        }
        results = []
        for future in futures:
            try:
                data = future.result()
                results.extend(data)
            except Exception as e:
                market = futures[future]
                logger.error(f"Failed to fetch data for market {market}: {e}")
    logger.info(f"Total records fetched: {len(results)}")
    return results

# Data Cleaning

def clean_entire_market_data(raw_data: List[OptionData]) -> pd.DataFrame:
    """
    Clean and transform raw option data into a structured DataFrame.

    Args:
        raw_data (List[OptionData]): The raw option data.

    Returns:
        pd.DataFrame: The cleaned and structured DataFrame.
    """
    logger.info("Starting data cleaning process.")
    df = pd.DataFrame(raw_data)

    # Identify general and specific columns
    general_columns = [col for col in df.columns if not (col.endswith("_P") or col.endswith("_C"))]
    specific_columns = [col.replace("_C", "") for col in df.columns if col.endswith("_C")]

    # Process Call Options
    calls = df[general_columns + [f"{col}_C" for col in specific_columns]]
    calls.columns = [col.replace("_C", "") for col in calls.columns]
    calls = calls.rename(columns=SPECIFIC_COLUMN_NAMES)
    calls['option_type'] = 'call'
    logger.debug("Processed call options.")

    # Process Put Options
    puts = df[general_columns + [f"{col}_P" for col in specific_columns]]
    puts.columns = [col.replace("_P", "") for col in puts.columns]
    puts = puts.rename(columns=SPECIFIC_COLUMN_NAMES)
    puts['option_type'] = 'put'
    logger.debug("Processed put options.")

    # Concatenate Calls and Puts
    result_df = pd.concat([calls, puts], ignore_index=True)
    result_df.rename(columns=GENERAL_COLUMN_NAMES, inplace=True)
    logger.info("Data cleaning completed.")
    return result_df

# Main Execution Flow

def fetch_cleaned_entire_market_data() -> pd.DataFrame:
    """
    Fetch and clean the entire option market data.

    Returns:
        pd.DataFrame: A DataFrame containing the cleaned option market data.
    """
    logger.info("Fetching entire market data.")
    raw_data = fetch_entire_market_data()
    logger.info("Cleaning fetched market data.")
    cleaned_data = clean_entire_market_data(raw_data)
    logger.info("Data fetched and cleaned successfully.")
    return cleaned_data

# Entry Point

if __name__ == "__main__":
    try:
        logger.info("Script started.")
        data = fetch_cleaned_entire_market_data()
        csv_path = "TSETMC_sample_data.csv"
        data.to_csv(path_or_buf=csv_path, index=False)
        logger.info(f"Data saved to {csv_path}.")

        # Display the first record for verification
        first_record = data.to_dict("records")[0]
        logger.info(f"First record: {first_record}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
