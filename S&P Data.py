import yfinance as yf
import datetime
import pandas as pd
import os

def get_sp500_data_to_file(years, file_path="sp500_data.csv"):
    """
    Fetches historical S&P 500 (GSPC) data from Yahoo Finance and saves it to a CSV file.

    Args:
        years (int): Number of years of historical data to fetch
        file_path (str): Path to save the CSV file (default: 'sp500_data.csv')

    Returns:
        pandas.DataFrame: The fetched data, or None if an error occurs
    """
    ticker_symbol = "^GSPC"  # Ticker for S&P 500
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=years * 365)

    print(f"Attempting to fetch S&P 500 data from Yahoo Finance for {years} years...")
    try:
        # Download data
        df = yf.download(ticker_symbol, start=start_date, end=end_date)

        if df.empty:
            print(
                f"Warning: No data fetched for {ticker_symbol} between {start_date} and {end_date}. Please check ticker symbol or date range.")
            return None

        # Reset index to make 'Date' a column
        df = df.reset_index()

        # Rename 'Date' column to 'Date' (it might be 'Datetime' from yfinance)
        df.rename(columns={'Date': 'Date'}, inplace=True)

        # Calculate daily percentage change
        df['Daily_Change_Percent'] = df['Close'].pct_change() * 100

        # Drop rows with NaN values (the first row for Daily_Change_Percent)
        df = df.dropna().reset_index(drop=True)

        # Save to CSV
        df.to_csv(file_path, index=False)
        print(f"Successfully fetched {len(df)} days of S&P 500 data and saved to {file_path}.")
        return df

    except Exception as e:
        print(f"Error fetching data from Yahoo Finance: {e}")
        return None


def load_sp500_data(file_path="sp500_data.csv"):
    """
    Loads S&P 500 data from a CSV file.

    Args:
        file_path (str): Path to the CSV file (default: 'sp500_data.csv')

    Returns:
        pandas.DataFrame: The loaded data, or None if an error occurs
    """
    try:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            return None

        df = pd.read_csv(file_path)
        # Ensure Date column is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"Successfully loaded S&P 500 data from {file_path}.")
        return df

    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

get_sp500_data_to_file(years=10,file_path="sp500_data.csv")
load_sp500_data(file_path="sp500_data.csv")