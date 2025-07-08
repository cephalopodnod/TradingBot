import pandas as pd
import os
import yfinance as yf
import datetime


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
        # Ensure Close and Daily_Change_Percent are float
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Daily_Change_Percent'] = pd.to_numeric(df['Daily_Change_Percent'], errors='coerce')
        # Check for missing or invalid values
        if df[['Date', 'Close', 'Daily_Change_Percent']].isna().any().any():
            print(f"Warning: Missing or invalid values found in {file_path}. Dropping rows with NaN.")
            df = df.dropna(subset=['Date', 'Close', 'Daily_Change_Percent'])
        print(f"Successfully loaded S&P 500 data from {file_path} with {len(df)} rows.")
        return df

    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def backtest_strategy(historical_data, buy_amount, drop_threshold):
    """
    Backtests the S&P 500 drop buying strategy.

    Args:
        historical_data (pd.DataFrame): DataFrame with 'Date', 'Close', and 'Daily_Change_Percent' columns.
        buy_amount (float): The amount to invest when the threshold is met.
        drop_threshold (float): The percentage drop (e.g., 1.0 for 1%) to trigger a buy.

    Returns:
        dict: A dictionary containing the simulation results.
    """
    if historical_data is None or historical_data.empty:
        print("Cannot run backtest: No historical data provided.")
        return {}

    # Verify required columns
    required_columns = ['Date', 'Close', 'Daily_Change_Percent']
    if not all(col in historical_data.columns for col in required_columns):
        print(f"Error: Historical data missing required columns: {required_columns}")
        return {}

    portfolio_value = 0.0
    total_invested = 0.0
    shares_held = 0.0
    buy_transactions = []
    daily_portfolio_values = []

    print(f"\n--- Starting Backtest Simulation ---")
    print(f"Strategy: Buy ${buy_amount} when S&P 500 drops by {drop_threshold}% daily.")
    print(
        f"Simulation Period: {historical_data['Date'].min().strftime('%Y-%m-%d')} to {historical_data['Date'].max().strftime('%Y-%m-%d')}\n")

    for index, row in historical_data.iterrows():
        current_date = row['Date']
        # Ensure current_price is a float
        try:
            current_price = float(row['Close'])
        except (TypeError, ValueError) as e:
            print(f"Error at index {index}, Date {current_date}: Invalid Close price {row['Close']}. Skipping row.")
            continue
        daily_change = row['Daily_Change_Percent']

        # Check for a drop meeting the threshold
        if daily_change <= -drop_threshold:
            shares_to_buy = buy_amount / current_price
            shares_held += shares_to_buy
            total_invested += buy_amount
            buy_transactions.append({
                'Date': current_date.strftime('%Y-%m-%d'),
                'Price_at_Buy': f"${current_price:,.2f}",
                'Daily_Change': f"{daily_change:,.2f}%",
                'Amount_Invested': f"${buy_amount:,.2f}",
                'Shares_Bought': f"{shares_to_buy:,.4f}"
            })

        # Update portfolio value based on current S&P price
        portfolio_value = shares_held * current_price
        daily_portfolio_values.append({'Date': current_date, 'Portfolio_Value': portfolio_value})

    # Final calculations
    final_sp_price = float(historical_data['Close'].iloc[-1])
    final_portfolio_value = shares_held * final_sp_price
    profit_loss = final_portfolio_value - total_invested
    roi = (profit_loss / total_invested) * 100 if total_invested > 0 else 0

    print(f"\n--- Backtest Results ---")
    print(f"Total historical data points: {len(historical_data)} trading days.")
    print(f"Number of buy transactions: {len(buy_transactions)}")
    print(f"Total Capital Invested: ${total_invested:,.2f}")
    print(f"Shares Held at End: {shares_held:,.4f}")
    print(f"Final S&P 500 Price: ${final_sp_price:,.2f}")
    print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
    print(f"Profit/Loss: ${profit_loss:,.2f}")
    print(f"Return on Investment (ROI): {roi:,.2f}%")

    # Display some transactions if available
    if buy_transactions:
        print("\n--- Sample Buy Transactions (first 5) ---")
        for i, tx in enumerate(buy_transactions[:5]):
            print(
                f"  {tx['Date']}: Bought {tx['Shares_Bought']} shares at {tx['Price_at_Buy']} (Daily Change: {tx['Daily_Change']}) for {tx['Amount_Invested']}")
        if len(buy_transactions) > 5:
            print(f"  ...and {len(buy_transactions) - 5} more transactions.")
    else:
        print("No buy transactions were triggered during the simulation period.")

    results = {
        'total_invested': total_invested,
        'final_portfolio_value': final_portfolio_value,
        'profit_loss': profit_loss,
        'roi': roi,
        'buy_transactions': buy_transactions,
        'daily_portfolio_values': pd.DataFrame(daily_portfolio_values)
    }
    return results


# Main execution
if __name__ == "__main__":
    # Load data from CSV
    data = load_sp500_data("sp500_data.csv")

    # Run backtest with specified settings
    if data is not None:
        # Debug: Print data info to verify
        print("DataFrame Info:")
        print(data.info())
        print("\nFirst few rows of data:")
        print(data.head())

        backtest_results = backtest_strategy(
            historical_data=data,
            buy_amount=100.0,  # $100 per buy
            drop_threshold=6  # 2% daily drop
        )