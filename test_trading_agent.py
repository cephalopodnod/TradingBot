import tempfile
import unittest
from pathlib import Path

import pandas as pd

from trading_agent import AlertManager, SP500DropModel, Signal, TradingAgent


class RecordingAlertManager(AlertManager):
    def __init__(self):
        super().__init__()
        self.messages = []

    def send(self, title: str, message: str) -> None:
        self.messages.append((title, message))


class TradingAgentTests(unittest.TestCase):
    def make_sp500_csv(self, directory: Path, drop_pct: float = -3.0) -> Path:
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        closes = [100.0, 98.0, 97.06]
        df = pd.DataFrame(
            {
                "Date": dates,
                "Close": closes,
                "Daily_Change_Percent": [None, -2.0, drop_pct],
            }
        )
        df = df.dropna(subset=["Daily_Change_Percent"]).reset_index(drop=True)
        path = directory / "sp500_test.csv"
        df.to_csv(path, index=False)
        return path

    def make_breakout_history(self) -> pd.DataFrame:
        close = [100 + (i * 2.0) for i in range(40)]
        return pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=40, freq="D"),
                "Close": close,
            }
        )

    def test_sp500_model_generates_signal(self):
        model = SP500DropModel(buy_amount=500.0, drop_threshold=2.0)
        data = pd.DataFrame(
            {
                "Close": [100.0, 98.0, 97.06],
                "Daily_Change_Percent": [-2.0, -2.0, -3.0],
            }
        )

        signals = model.evaluate(data, "^GSPC")

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, "buy")
        self.assertEqual(signals[0].symbol, "^GSPC")
        self.assertAlmostEqual(signals[0].quantity, 500.0 / 97.06, places=4)

    def test_paper_trade_acceptance_updates_snapshot_and_ledger(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            alert_manager = RecordingAlertManager()
            agent = TradingAgent(
                data_dir=data_dir,
                alert_manager=alert_manager,
                paper_starting_cash=1000.0,
                paper_ledger_path="paper.json",
                live_ledger_path="live.json",
            )

            signal = Signal(
                model_name="SP500DropModel",
                symbol="^GSPC",
                action="buy",
                reason="Test signal",
                confidence=0.8,
                price=50.0,
                quantity=2.0,
                account="paper",
            )

            record = agent.accept_trade(signal, account="paper")

            snapshot = agent.get_account_snapshot("paper")
            self.assertEqual(record.status, "accepted")
            self.assertEqual(snapshot["cash"], 900.0)
            self.assertEqual(snapshot["positions"]["^GSPC"]["quantity"], 2.0)
            self.assertEqual(len(agent.get_trade_history("paper")), 1)

    def test_run_sp500_scan_sends_alert(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            alert_manager = RecordingAlertManager()
            agent = TradingAgent(data_dir=data_dir, alert_manager=alert_manager)
            csv_path = self.make_sp500_csv(data_dir)

            signals = agent.run_sp500_scan(csv_path)

            self.assertEqual(len(signals), 1)
            self.assertEqual(len(alert_manager.messages), 1)
            self.assertIn("Trade Alert", alert_manager.messages[0][0])
            self.assertIn("S&P 500 dropped", alert_manager.messages[0][1])

    def test_profit_target_scan_creates_sell_signal_and_updates_paper_account(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            alert_manager = RecordingAlertManager()
            agent = TradingAgent(
                data_dir=data_dir,
                alert_manager=alert_manager,
                paper_starting_cash=1000.0,
                paper_ledger_path="paper.json",
                live_ledger_path="live.json",
            )

            buy_signal = Signal(
                model_name="SP500DropModel",
                symbol="AAPL",
                action="buy",
                reason="Test buy",
                confidence=0.8,
                price=100.0,
                quantity=1.0,
                account="paper",
            )
            agent.accept_trade(buy_signal, account="paper")

            market_data = {
                "AAPL": pd.DataFrame(
                    {
                        "Close": [100.0, 106.0, 107.5],
                    }
                )
            }

            signals = agent.run_profit_target_scan(account="paper", market_data_by_symbol=market_data)
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0].action, "sell")
            self.assertEqual(signals[0].symbol, "AAPL")

            record = agent.accept_trade(signals[0], account="paper")

            snapshot = agent.get_account_snapshot("paper")
            self.assertEqual(record.status, "accepted")
            self.assertEqual(snapshot["cash"], 1007.5)
            self.assertNotIn("AAPL", snapshot["positions"])

    def test_analyze_ticker_for_model_returns_a_suggestion(self):
        agent = TradingAgent()
        suggestion = agent.analyze_ticker_for_model("FAKE", historical_data=self.make_breakout_history())

        self.assertEqual(suggestion.symbol, "FAKE")
        self.assertGreater(suggestion.score, 0.0)
        self.assertIn(suggestion.model_name, {"MomentumBreakout", "MovingAverageCross", "PullbackAfterMomentum"})

    def test_accept_custom_model_and_scan_custom_model_alerts_for_buy_and_sell(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            alert_manager = RecordingAlertManager()
            agent = TradingAgent(
                data_dir=data_dir,
                alert_manager=alert_manager,
                paper_starting_cash=1000.0,
                paper_ledger_path="paper.json",
                live_ledger_path="live.json",
            )

            breakout_history = self.make_breakout_history()
            suggestion = agent.analyze_ticker_for_model("FAKE", historical_data=breakout_history)
            config = agent.accept_custom_model("FAKE", suggestion, allocation_amount=250.0, account="paper")

            self.assertEqual(config.symbol, "FAKE")
            self.assertEqual(config.allocation_amount, 250.0)

            buy_signals = agent.scan_custom_model("FAKE", account="paper", historical_data=breakout_history)
            self.assertEqual(len(buy_signals), 1)
            self.assertEqual(buy_signals[0].action, "buy")

            agent.accept_trade(buy_signals[0], account="paper")
            self.assertEqual(agent.get_account_snapshot("paper")["positions"]["FAKE"]["quantity"], buy_signals[0].quantity)

            sell_history = pd.DataFrame(
                {
                    "Date": pd.date_range("2020-01-01", periods=40, freq="D"),
                    "Close": [200.0] * 40,
                }
            )

            signals = agent.scan_custom_model("FAKE", account="paper", historical_data=sell_history)

            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0].action, "sell")
            self.assertEqual(signals[0].symbol, "FAKE")
            self.assertEqual(len(alert_manager.messages), 2)
            self.assertIn("sell", alert_manager.messages[-1][1].lower())

    def test_backtest_ticker_models_returns_summary_for_selected_models(self):
        agent = TradingAgent()

        history = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=60, freq="D"),
                "Close": [100.0] * 20
                + [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0]
                + [111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0, 140.0],
            }
        )

        results = agent.backtest_ticker_models(
            "FAKE",
            historical_data=history,
            selected_models=["MovingAverageCross", "MomentumBreakout", "PullbackAfterMomentum"],
            initial_cash=1000.0,
        )

        self.assertEqual(len(results), 3)
        self.assertTrue(all("model_name" in result for result in results))
        self.assertTrue(all("final_cash" in result for result in results))
        self.assertTrue(all("trade_count" in result for result in results))

    def test_backtest_ticker_models_supports_dividend_capture(self):
        agent = TradingAgent()
        now = pd.Timestamp.now("UTC")
        history = pd.DataFrame(
            {
                "Date": pd.date_range(now - pd.Timedelta(days=30), periods=30, freq="D"),
                "Close": [100.0] * 30,
            }
        )

        results = agent.backtest_ticker_models(
            "FAKE",
            historical_data=history,
            selected_models=["DividendCaptureModel"],
            initial_cash=1000.0,
            dividend_data_by_symbol={"FAKE": pd.DataFrame({"Date": [now + pd.Timedelta(days=5)], "Dividends": [1.5]})},
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["model_name"], "DividendCaptureModel")
        self.assertGreaterEqual(results[0]["final_cash"], 1000.0)

    def test_scan_model_handles_dividend_models(self):
        agent = TradingAgent()
        now = pd.Timestamp.now("UTC")
        history = pd.DataFrame(
            {
                "Date": pd.date_range(now - pd.Timedelta(days=30), periods=30, freq="D"),
                "Close": [100.0] * 30,
            }
        )
        dividend_data = pd.DataFrame({"Date": [now + pd.Timedelta(days=5)], "Dividends": [1.5]})

        dividend_signals = agent.scan_model("FAKE", "DividendCaptureModel", historical_data=history, dividend_data=dividend_data, allocation_amount=250.0)

        self.assertEqual(dividend_signals[0].model_name, "DividendCaptureModel")

    def test_add_ticker_position_attaches_algorithm_and_records_alert(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            alert_manager = RecordingAlertManager()
            agent = TradingAgent(
                data_dir=data_dir,
                alert_manager=alert_manager,
                paper_starting_cash=1000.0,
                paper_ledger_path="paper.json",
                live_ledger_path="live.json",
            )
            history = pd.DataFrame(
                {
                    "Date": pd.date_range("2024-01-01", periods=30, freq="D"),
                    "Close": [100.0] * 30,
                }
            )

            config = agent.add_ticker_position(
                symbol="FAKE",
                model_name="MovingAverageCross",
                allocation_amount=250.0,
                account="paper",
                historical_data=history,
            )

            self.assertEqual(config.symbol, "FAKE")
            self.assertEqual(config.model_name, "MovingAverageCross")
            self.assertEqual(config.allocation_amount, 250.0)
            self.assertEqual(agent.get_account_snapshot("paper")["positions"]["FAKE"]["quantity"], 2.5)
            self.assertEqual(len(agent.get_trade_history("paper")), 1)
            self.assertEqual(len(alert_manager.messages), 1)

    def test_add_bank_transaction_updates_performance_and_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            agent = TradingAgent(
                data_dir=data_dir,
                paper_starting_cash=1000.0,
                paper_ledger_path="paper.json",
                live_ledger_path="live.json",
            )

            record = agent.add_bank_transaction(500.0, account="paper", notes="Monthly transfer")

            snapshot = agent.get_account_snapshot("paper")
            performance = agent.get_account_performance("paper")
            timeline = agent.get_account_timeline("paper")

            self.assertEqual(record.action, "deposit")
            self.assertEqual(snapshot["cash"], 1500.0)
            self.assertEqual(performance["total_deposits"], 500.0)
            self.assertGreater(performance["benchmark_projection"], 500.0)
            self.assertEqual(len(timeline), 1)
            self.assertEqual(len(agent.get_trade_history("paper")), 1)

    def test_backtest_dividend_capture_returns_strategy_comparison(self):
        agent = TradingAgent()
        now = pd.Timestamp.now("UTC")
        history = pd.DataFrame(
            {
                "Date": pd.date_range(now - pd.Timedelta(days=24), periods=25, freq="D"),
                "Close": [100.0, 99.8, 99.6, 99.3, 99.1, 98.8, 98.5, 98.0, 97.7, 97.3, 97.0, 96.8, 96.5, 96.0, 95.8, 95.5, 95.2, 95.0, 94.8, 94.6, 94.3, 94.0, 93.8, 93.5, 93.0],
            }
        )
        dividend_data = pd.DataFrame(
            {
                "Date": [now + pd.Timedelta(days=5)],
                "Dividends": [1.5],
            }
        )

        results = agent.backtest_dividend_capture(
            ["FAKE"],
            historical_data_by_symbol={"FAKE": history},
            dividend_data_by_symbol={"FAKE": dividend_data},
            dividend_window=45,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["symbol"], "FAKE")
        self.assertIn("hold_is_better", results[0])
        self.assertIn("recommended_action", results[0])

    def test_run_dividend_capture_scan_returns_rotation_signals(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            alert_manager = RecordingAlertManager()
            agent = TradingAgent(
                data_dir=data_dir,
                alert_manager=alert_manager,
                paper_starting_cash=1000.0,
                paper_ledger_path="paper.json",
                live_ledger_path="live.json",
            )
            now = pd.Timestamp.now("UTC")
            old_history = pd.DataFrame(
                {
                    "Date": pd.date_range(now - pd.Timedelta(days=24), periods=25, freq="D"),
                    "Close": [100.0, 99.8, 99.6, 99.3, 99.1, 98.8, 98.5, 98.0, 97.7, 97.3, 97.0, 96.8, 96.5, 96.0, 95.8, 95.5, 95.2, 95.0, 94.8, 94.6, 94.3, 94.0, 93.8, 93.5, 93.0],
                }
            )
            new_history = pd.DataFrame(
                {
                    "Date": pd.date_range(now - pd.Timedelta(days=24), periods=25, freq="D"),
                    "Close": [100.0] * 25,
                }
            )
            old_dividends = pd.DataFrame(
                {
                    "Date": [now + pd.Timedelta(days=12)],
                    "Dividends": [1.0],
                }
            )
            new_dividends = pd.DataFrame(
                {
                    "Date": [now + pd.Timedelta(days=5)],
                    "Dividends": [1.5],
                }
            )

            agent.accept_trade(
                Signal(
                    model_name="DividendCaptureModel",
                    symbol="OLD",
                    action="buy",
                    reason="Seed dividend position",
                    confidence=0.9,
                    price=95.0,
                    quantity=2.0,
                    account="paper",
                ),
                account="paper",
            )

            signals = agent.run_dividend_capture_scan(
                ["OLD", "NEW"],
                account="paper",
                allocation_amount=200.0,
                historical_data_by_symbol={"OLD": old_history, "NEW": new_history},
                dividend_data_by_symbol={"OLD": old_dividends, "NEW": new_dividends},
            )

            self.assertTrue(any(signal.action == "sell" and signal.symbol == "OLD" for signal in signals))
            self.assertTrue(any(signal.action == "buy" and signal.symbol == "NEW" for signal in signals))
            self.assertGreaterEqual(len(alert_manager.messages), 2)


if __name__ == "__main__":
    unittest.main()
