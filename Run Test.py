import argparse
import os

import numpy as np
import pandas as pd

from trading_agent import TradingAgent


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
        df["Date"] = pd.to_datetime(df["Date"])
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["Daily_Change_Percent"] = pd.to_numeric(df["Daily_Change_Percent"], errors="coerce")

        if df[["Date", "Close", "Daily_Change_Percent"]].isna().any().any():
            print(f"Warning: Missing or invalid values found in {file_path}. Dropping rows with NaN.")
            df = df.dropna(subset=["Date", "Close", "Daily_Change_Percent"])

        print(f"Successfully loaded S&P 500 data from {file_path} with {len(df)} rows.")
        return df

    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def backtest_strategy(historical_data, buy_amount, buy_threshold, sell_threshold=0.0, verbose=True):
    """
    Backtests a simple buy/sell strategy based on daily percentage change.

    Args:
        historical_data (pd.DataFrame): DataFrame with 'Date', 'Close', and 'Daily_Change_Percent' columns.
        buy_amount (float): The amount to invest when the buy threshold is met.
        buy_threshold (float): The percentage drop to trigger a buy.
        sell_threshold (float): The percentage rise to trigger a sell.

    Returns:
        dict: A dictionary containing the simulation results.
    """
    if historical_data is None or historical_data.empty:
        print("Cannot run backtest: No historical data provided.")
        return {}

    required_columns = ["Date", "Close", "Daily_Change_Percent"]
    if not all(col in historical_data.columns for col in required_columns):
        print(f"Error: Historical data missing required columns: {required_columns}")
        return {}

    cash = 0.0
    shares_held = 0.0
    total_invested = 0.0
    buy_transactions = []
    sell_transactions = []
    portfolio_history = []

    if verbose:
        print(f"\n--- Starting Backtest Simulation ---")
        print(f"Strategy: Buy ${buy_amount} when S&P 500 drops by {buy_threshold}%.")
        if sell_threshold > 0.0:
            print(f"Sell when S&P 500 rises by {sell_threshold}% after a position is open.")
        print(
            f"Simulation Period: {historical_data['Date'].min().strftime('%Y-%m-%d')} to {historical_data['Date'].max().strftime('%Y-%m-%d')}\n"
        )

    for index, row in historical_data.iterrows():
        current_date = row["Date"]
        try:
            current_price = float(row["Close"])
        except (TypeError, ValueError):
            print(f"Error at index {index}, Date {current_date}: Invalid Close price {row['Close']}. Skipping row.")
            continue

        daily_change = float(row["Daily_Change_Percent"])

        if shares_held <= 0.0 and daily_change <= -buy_threshold:
            shares_to_buy = buy_amount / current_price
            shares_held += shares_to_buy
            cash -= buy_amount
            total_invested += buy_amount
            buy_transactions.append(
                {
                    "Date": current_date.strftime("%Y-%m-%d"),
                    "Price_at_Buy": f"${current_price:,.2f}",
                    "Daily_Change": f"{daily_change:,.2f}%",
                    "Amount_Invested": f"${buy_amount:,.2f}",
                    "Shares_Bought": f"{shares_to_buy:,.4f}",
                }
            )

        elif shares_held > 0.0 and sell_threshold > 0.0 and daily_change >= sell_threshold:
            proceeds = shares_held * current_price
            cash += proceeds
            sell_transactions.append(
                {
                    "Date": current_date.strftime("%Y-%m-%d"),
                    "Price_at_Sell": f"${current_price:,.2f}",
                    "Daily_Change": f"{daily_change:,.2f}%",
                    "Shares_Sold": f"{shares_held:,.4f}",
                    "Proceeds": f"${proceeds:,.2f}",
                }
            )
            shares_held = 0.0

        portfolio_value = cash + shares_held * current_price
        portfolio_history.append({"Date": current_date, "Portfolio_Value": portfolio_value})

    final_price = float(historical_data["Close"].iloc[-1])
    final_portfolio_value = cash + shares_held * final_price
    profit_loss = final_portfolio_value - total_invested
    roi = (profit_loss / total_invested) * 100 if total_invested > 0 else 0

    if verbose:
        print(f"\n--- Backtest Results ---")
        print(f"Total historical data points: {len(historical_data)} trading days.")
        print(f"Total buy transactions: {len(buy_transactions)}")
        print(f"Total sell transactions: {len(sell_transactions)}")
        print(f"Total Capital Invested: ${total_invested:,.2f}")
        print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
        print(f"Profit/Loss: ${profit_loss:,.2f}")
        print(f"Return on Investment (ROI): {roi:,.2f}%")

        if buy_transactions:
            print("\n--- Sample Buy Transactions (first 5) ---")
            for tx in buy_transactions[:5]:
                print(
                    f"  {tx['Date']}: Bought {tx['Shares_Bought']} shares at {tx['Price_at_Buy']} (Daily Change: {tx['Daily_Change']}) for {tx['Amount_Invested']}"
                )
            if len(buy_transactions) > 5:
                print(f"  ...and {len(buy_transactions) - 5} more transactions.")

        if sell_transactions:
            print("\n--- Sample Sell Transactions (first 5) ---")
            for tx in sell_transactions[:5]:
                print(
                    f"  {tx['Date']}: Sold {tx['Shares_Sold']} shares at {tx['Price_at_Sell']} (Daily Change: {tx['Daily_Change']}) for {tx['Proceeds']}"
                )
            if len(sell_transactions) > 5:
                print(f"  ...and {len(sell_transactions) - 5} more transactions.")

    return {
        "total_invested": total_invested,
        "final_portfolio_value": final_portfolio_value,
        "profit_loss": profit_loss,
        "roi": roi,
        "buy_transactions": buy_transactions,
        "sell_transactions": sell_transactions,
        "portfolio_history": pd.DataFrame(portfolio_history),
    }


def optimize_thresholds(historical_data, buy_amount, buy_range, sell_range, step):
    """
    Grid-search buy and sell thresholds to maximize ROI.

    Args:
        historical_data (pd.DataFrame): DataFrame with daily percentage changes.
        buy_amount (float): Fixed buy allocation per signal.
        buy_range (tuple[float, float]): (min, max) buy drop percentages.
        sell_range (tuple[float, float]): (min, max) sell rise percentages.
        step (float): Search step size for thresholds.

    Returns:
        dict: Best thresholds and the associated results.
    """
    best_result = None
    scan_results = []
    buy_start, buy_end = buy_range
    sell_start, sell_end = sell_range

    buy_values = [round(x, 4) for x in np.arange(buy_start, buy_end + step / 2, step)]
    sell_values = [round(x, 4) for x in np.arange(sell_start, sell_end + step / 2, step)]

    for buy_threshold in buy_values:
        for sell_threshold in sell_values:
            result = backtest_strategy(historical_data, buy_amount, buy_threshold, sell_threshold, verbose=False)
            if not result:
                continue
            score = result.get("roi", float("-inf"))
            scan_results.append((buy_threshold, sell_threshold, score, result))
            if best_result is None or score > best_result[2]:
                best_result = (buy_threshold, sell_threshold, score, result)

    if best_result is None:
        raise RuntimeError("Optimization could not find any valid threshold combination.")

    buy_threshold, sell_threshold, best_score, best_data = best_result
    print(f"\nOptimization complete: best ROI {best_score:.2f}% at buy threshold {buy_threshold}% and sell threshold {sell_threshold}%.")
    return {
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "roi": best_score,
        "result": best_data,
        "scan_count": len(scan_results),
    }


def prompt_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print("Invalid input. Using default value.")
        return default


def print_latest_quote(symbol: str) -> None:
    agent = TradingAgent()
    try:
        quote = agent.fetch_latest_quote(symbol)
        print(f"Latest quote for {symbol}: ${quote.price:.2f}")
        print(f"Timestamp: {quote.timestamp.isoformat()}")
        print(f"Delay: {quote.delay_seconds:.1f} seconds")
        print(f"Source: {quote.source}")
    except Exception as exc:
        print(f"Unable to fetch quote for {symbol}: {exc}")


def run_gui() -> None:
    from trading_gui import launch_trading_gui

    print("Launching Trading Bot GUI...")
    launch_trading_gui()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Trading Bot app or backtest utilities.")
    parser.add_argument("--gui", action="store_true", help="Launch the trading bot GUI.")
    parser.add_argument("--ticker", type=str, help="Fetch the latest quote for a specific ticker symbol.")
    parser.add_argument("--csv", type=str, default="sp500_data.csv", help="Path to the CSV file for backtesting.")
    parser.add_argument("--buy-amount", type=float, default=100.0, help="Amount to invest per buy signal.")
    parser.add_argument("--buy-threshold", "--drop-threshold", type=float, default=3.5, help="Daily percentage drop threshold to trigger a buy.")
    parser.add_argument("--sell-threshold", type=float, default=10.0, help="Daily percentage rise threshold to trigger a sell.")
    parser.add_argument("--optimize", action="store_true", help="Search for the best buy and sell thresholds to maximize ROI.")
    parser.add_argument("--buy-range", type=float, nargs=2, default=[1.0, 10.0], help="Min and max buy threshold values for optimization.")
    parser.add_argument("--sell-range", type=float, nargs=2, default=[0.0, 10.0], help="Min and max sell threshold values for optimization.")
    parser.add_argument("--step", type=float, default=0.5, help="Step size for threshold optimization search.")
    args = parser.parse_args()

    if args.gui:
        run_gui()
        return

    if args.ticker:
        print_latest_quote(args.ticker.upper())
        return

    data = load_sp500_data(args.csv)
    if data is None:
        return

    buy_threshold = args.buy_threshold
    sell_threshold = args.sell_threshold

    if not args.optimize and args.buy_threshold == 6.0 and args.sell_threshold == 0.0:
        buy_threshold = prompt_float("Enter buy drop threshold percent", buy_threshold)
        sell_threshold = prompt_float("Enter sell rise threshold percent", sell_threshold)

    print("DataFrame Info:")
    print(data.info())
    print("\nFirst few rows of data:")
    print(data.head())

    if args.optimize:
        optimization = optimize_thresholds(
            historical_data=data,
            buy_amount=args.buy_amount,
            buy_range=tuple(args.buy_range),
            sell_range=tuple(args.sell_range),
            step=args.step,
        )
        print(f"Best buy threshold: {optimization['buy_threshold']}%")
        print(f"Best sell threshold: {optimization['sell_threshold']}%")
        print(f"Best ROI: {optimization['roi']:.2f}%")
        print(f"Scanned {optimization['scan_count']} threshold combinations.")
        return
        print(f"Best buy threshold: {optimization['buy_threshold']}%")
        print(f"Best sell threshold: {optimization['sell_threshold']}%")
        print(f"Best ROI: {optimization['roi']:.2f}%")
        print(f"Scanned {optimization['scan_count']} threshold combinations.")
    else:
        backtest_strategy(
            historical_data=data,
            buy_amount=args.buy_amount,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )


if __name__ == "__main__":
    main()
