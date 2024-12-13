import os
from dotenv import load_dotenv
import warnings
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import ssl
import certifi
import time

warnings.filterwarnings('ignore')

def get_data(company_ticker, startyear, endyear):

    # Setting parameters
    company_tick = company_ticker
    start_year = startyear
    end_year = endyear

    load_dotenv('.env') # Load environment variables from the file '.env'
    api_key = os.environ.get('API_KEY') # Retrieve API key value
    api_key_fred = os.environ.get('API_KEY_FRED') # Retrieve API FRED key value

    base_url = 'https://financialmodelingprep.com/api/v3'
    endpoint_url = f"{base_url}/historical-price-full/stock_dividend/{company_tick}?apikey={api_key}"

    response = requests.get(endpoint_url)
    if response.status_code == 429:
        print("FMP API limit reached")
        return 0

    # Convert json to dictionary object and then a Pandas Dataframe
    response_dict = response.json()
    dividends = pd.DataFrame(response_dict['historical'])

    # Define our preferred date range
    end_year = end_year + 1

    # Data transformations
    if dividends.shape == (0,0): # Case when company has never given out dividends
        dividends = pd.DataFrame(
            {"year": list(range(start_year - 1, end_year+1)),
             "adjDividend": [0] * len(list(range(start_year - 1, end_year + 1)))
             })
    else:
        # Extract year data from date column
        dividends['year'] = pd.to_datetime(dividends['date']).dt.year
        # Aggregate the dividends paid by year
        dividends = dividends.groupby("year").agg({"adjDividend" : "sum"}).reset_index()
        # Create new dataframe with all years from start to end
        all_years = pd.DataFrame({"year" : list(range(start_year - 1, end_year + 1))})
        # Merge the two dataframes on the year column and fill missing values with 0.0
        dividends = all_years.merge(dividends, on = "year", how = "left").fillna(0.0)

    # Create target variable
    dividends["next_year_dividend"] = dividends["adjDividend"].shift(-1)

    conditions = [
        dividends["adjDividend"] <= dividends["next_year_dividend"],
        dividends["adjDividend"] > dividends["next_year_dividend"]
    ]

    choices = ["constant/increased", "decreased"]

    # Create the target column based on the change to dividend
    dividends["dps_trend"] = np.select(conditions, choices, default = "n/a")

    # Create new features
    dividends["last_year_dividend"] = dividends["adjDividend"].shift(1)
    dividends["dps_growth"] = dividends["adjDividend"] - dividends["last_year_dividend"]

    dividends["dps_growth_rate"] = np.where(
        (dividends["last_year_dividend"] == 0) & (dividends["adjDividend"] == 0), 0,
        np.where(
            dividends["last_year_dividend"] != 0,
            ((dividends["adjDividend"] / dividends["last_year_dividend"]) - 1) * 100,
            999
        )
    )

    # Remove first year (NaN)
    dividends = dividends.loc[(dividends["year"] >= start_year) & (dividends["year"] <= end_year - 1)]

    # Only keep columns needed
    dividends = dividends[["year", "adjDividend", "dps_growth", "dps_growth_rate","dps_trend"]]

    # Engineering other predictors
    predictors = pd.DataFrame({"year" : list(range(start_year - 1, end_year))})

    # Let's include the Company's Industry and sector data
    company_data_raw = yf.Ticker(company_tick)
    company_data = company_data_raw.info
    if 'industry' in company_data:
        predictors["industry"] = company_data['industry']
    else:
        print(f"Company {company_tick} data is not available")
        return 0
    if 'sector' in company_data:
        predictors["sector"] = company_data['sector']
    else:
        print(f"Company {company_tick} data is not available")
        return 0

    # Adding macroeconomic information from FRED

    # 1. Federal Interest Rate
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&' \
        f'api_key={api_key_fred}&' \
        f'file_type=json&' \
        f'observation_start={str(start_year - 1) + "-01-01"}&observation_end={str(end_year -1) + "-12-31"}&' \
        f'frequency=a'

    response = requests.get(url)
    fed_interest_rates = pd.DataFrame(response.json()["observations"])
    predictors["interest_rate"] = fed_interest_rates["value"].astype("float64")

    # 2. Market Inflation Rate
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id=FPCPITOTLZGUSA&' \
        f'api_key={api_key_fred}&' \
        f'file_type=json&' \
        f'observation_start={str(start_year - 1) + "-01-01"}&observation_end={str(end_year -1) + "-12-31"}&' \
        f'frequency=a'

    response = requests.get(url)
    inflation_rates = pd.DataFrame(response.json()["observations"])
    predictors["inflation_rate"] = inflation_rates["value"].astype("float64")

    # 3. US GDP
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id=GDP&' \
        f'api_key={api_key_fred}&' \
        f'file_type=json&' \
        f'observation_start={str(start_year - 1) + "-01-01"}&observation_end={str(end_year -1) + "-12-31"}&' \
        f'frequency=a'

    response = requests.get(url)
    GDP_values = pd.DataFrame(response.json()["observations"])
    predictors["GDP"] = GDP_values["value"].astype("float64")

    # 4. Highest Marginal Tax Rate - decommissioned
    #url = f'https://api.stlouisfed.org/fred/series/observations?series_id=IITTRHB&' \
    #    f'api_key={api_key_fred}&' \
    #    f'file_type=json&' \
    #    f'observation_start={str(start_year - 1) + "-01-01"}&observation_end={str(end_year -1) + "-12-31"}&' \
    #    f'frequency=a'

    #response = requests.get(url)
    #highest_tax_rates = pd.DataFrame(response.json()["observations"])
    #predictors["highest_tax_rate"] = highest_tax_rates["value"].astype("float64")

    # 5. Lowest Marginal Tax Rate - decommissioned
    # url = f'https://api.stlouisfed.org/fred/series/observations?series_id=IITTRLB&' \
    #    f'api_key={api_key_fred}&' \
    #    f'file_type=json&' \
    #    f'observation_start={str(start_year - 1) + "-01-01"}&observation_end={str(end_year -1) + "-12-31"}&' \
    #    f'frequency=a'

    #response = requests.get(url)
    #lowest_tax_rates = pd.DataFrame(response.json()["observations"])
    #predictors["lowest_tax_rate"] = lowest_tax_rates["value"].astype("float64")

    # 6. Difference in Tax Rates - decommissioned
    # predictors["tax_rate_difference"] = predictors["highest_tax_rate"] - predictors["lowest_tax_rate"]

    # 7. US Unemployment Rate
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id=UNRATE&' \
        f'api_key={api_key_fred}&' \
        f'file_type=json&' \
        f'observation_start={str(start_year - 1) + "-01-01"}&observation_end={str(end_year -1) + "-12-31"}&' \
        f'frequency=a'

    response = requests.get(url)
    unemployment_rates = pd.DataFrame(response.json()["observations"])
    predictors["Unemployment Rate"] = unemployment_rates["value"].astype("float64")

    # 8. Consumer Sentiment Rate (per University of Michigan)
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id=UMCSENT&' \
        f'api_key={api_key_fred}&' \
        f'file_type=json&' \
        f'observation_start={str(start_year - 1) + "-01-01"}&observation_end={str(end_year -1) + "-12-31"}&' \
        f'frequency=a'

    response = requests.get(url)
    consumer_sentiment_rates = pd.DataFrame(response.json()["observations"])
    predictors["Consumer Sentiment Rate"] = consumer_sentiment_rates["value"].astype("float64")

    # Adding other financial ratios
    num_of_years = 2024 - start_year + 1

    response = requests.get(f"{base_url}/ratios/{company_tick}?limit={num_of_years}&apikey={api_key}")

    # Check if all year's data is available
    data_length = len(response.json())
    if data_length != num_of_years:
        print(f"Company {company_tick} data is not available.")
        return 0

    financial_ratios = pd.DataFrame(response.json()).iloc[:,:].sort_values("date", ascending = True).reset_index(drop=True)
    financial_ratios["calendarYear"] = financial_ratios["calendarYear"].astype("int64")
    predictors = predictors.merge(financial_ratios, left_on = "year", right_on = "calendarYear", how = "left").fillna(0.0)

    # Drop unnecessary columns
    predictors.drop(["date","calendarYear","period"], axis = "columns", inplace = True)

    def calculate_change(df, feature_name):
        # Calculating pct change on new predictors
        percent_change = df[feature_name].pct_change() * 100
        # Create new column name
        new_col_name = f"{feature_name}_percent_change"
        # Find column position of original predictor
        original_col_position = df.columns.get_loc(feature_name)
        # Insert new column to right of above column
        df.insert(original_col_position + 1, new_col_name, percent_change)

    feature_list = list(predictors.columns)
    feature_list.remove("year")
    feature_list.remove("sector")
    feature_list.remove("industry")
    feature_list.remove("symbol")

    for feature in feature_list:
        calculate_change(predictors, feature)

    # Replacing inf and NaN values
    predictors.replace([float('inf'), float('-inf')], 999, inplace = True)
    predictors.fillna(0, inplace = True)

    # Combine dividend data with other predictors
    dataset = pd.merge(dividends, predictors, left_on='year', right_on='year', how='left')

    # Move target to the end of the dataset for good practice
    feature_list = list(dataset.columns)
    feature_list.append('dps_change_next_year')
    feature_list.remove('dps_change_next_year')
    dataset = dataset[feature_list]

    return dataset

apple_data = get_data('AAPL', 2013, 2022)


import urllib.request


# Use our data extractor to build our database by scraping Wikipedia
wikiurl = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
response = requests.get(wikiurl, verify=certifi.where())  # Use certifi for SSL verification

# Check if the request was successful
if response.status_code == 200:
    # Parse the tables using pandas
    tables = pd.read_html(response.text)
    ticker_table = tables[0]  # Extract the first table
    tickers = ticker_table['Symbol'].tolist()
else:
    print(f"Failed to fetch URL. Status code: {response.status_code}")

# ------------------------------------------------------------------

start_year = 2013
end_year = 2022

dataset = []
company_number = 1

for ticker in tickers:
    print(f"{company_number}: Obtaining data for {ticker}")
    company_number = company_number + 1
    company_data = get_data(ticker, start_year, end_year)
    if type(company_data).__name__ == "int":
        continue
    dataset.append(company_data)
    time.sleep(3)  # Add a 1-second delay between requests
dataset = pd.concat(dataset, ignore_index=True)

# Save data to disk
dataset.to_csv("Stock_data.csv", index=False)