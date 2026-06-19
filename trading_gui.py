from __future__ import annotations

from tkinter import BooleanVar, Canvas, StringVar, Tk, messagebox
from tkinter.scrolledtext import ScrolledText
from tkinter.ttk import Button, Checkbutton, Combobox, Entry, Frame, Label, Notebook, Style

import pandas as pd

from trading_agent import AlertManager, CustomModelConfig, TradingAgent


class GuiAlertManager(AlertManager):
    def __init__(self, window):
        super().__init__()
        self.window = window

    def send(self, title: str, message: str) -> None:
        self.window.append_log(f"[{title}] {message}")


class TradingBotWindow:
    AVAILABLE_MODELS = ["MovingAverageCross", "MomentumBreakout", "PullbackAfterMomentum", "DividendCaptureModel"]

    def __init__(self, root: Tk | None = None):
        self.root = root or Tk()
        self.root.title("Trading Bot Workspace")
        self.root.geometry("980x760")
        self.root.minsize(920, 700)

        self.alert_manager = GuiAlertManager(self)
        self.agent = TradingAgent(alert_manager=self.alert_manager)

        self.current_suggestion = None
        self.current_symbol = ""
        self.pending_signals = []
        self.last_history: pd.DataFrame | None = None
        self.last_backtest = []
        self.best_model_name = ""

        self.model_vars = {name: BooleanVar(value=True) for name in self.AVAILABLE_MODELS}
        self.ticker_account = StringVar(value="paper")

        self._build_ui()
        self.refresh_account()

    def _build_ui(self) -> None:
        style = Style()
        style.theme_use("clam")

        main_frame = Frame(self.root, padding=12)
        main_frame.pack(fill="both", expand=True)

        title = Label(main_frame, text="Trading Bot", font=("Segoe UI", 18, "bold"))
        title.pack(anchor="w", pady=(0, 10))

        self.notebook = Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True)

        self.account_tab = Frame(self.notebook)
        self.ticker_tab = Frame(self.notebook)
        self.trade_tab = Frame(self.notebook)
        self.algorithm_tab = Frame(self.notebook)

        self.notebook.add(self.account_tab, text="Account")
        self.notebook.add(self.ticker_tab, text="Ticker Search")
        self.notebook.add(self.trade_tab, text="Trade")
        self.notebook.add(self.algorithm_tab, text="Algorithm Builder")

        self._build_account_tab()
        self._build_ticker_tab()
        self._build_trade_tab()
        self._build_algorithm_tab()

    def _build_account_tab(self) -> None:
        frame = Frame(self.account_tab, padding=12)
        frame.pack(fill="both", expand=True)

        self.cash_label = Label(frame, text="Cash: $0.00", font=("Segoe UI", 12, "bold"))
        self.cash_label.pack(anchor="w")

        self.positions_label = Label(frame, text="Positions: none", font=("Segoe UI", 10))
        self.positions_label.pack(anchor="w", pady=(6, 12))

        history_label = Label(frame, text="Trade history", font=("Segoe UI", 12, "bold"))
        history_label.pack(anchor="w")

        self.history_text = ScrolledText(frame, height=14, wrap="word")
        self.history_text.pack(fill="both", expand=True, pady=(6, 0))
        self.history_text.configure(state="disabled")

        refresh_button = Button(frame, text="Refresh", command=self.refresh_account)
        refresh_button.pack(anchor="e", pady=(8, 0))

    def _build_ticker_tab(self) -> None:
        frame = Frame(self.ticker_tab, padding=12)
        frame.pack(fill="both", expand=True)

        Label(frame, text="Ticker search", font=("Segoe UI", 12, "bold")).pack(anchor="w")

        search_row = Frame(frame)
        search_row.pack(fill="x", pady=(8, 8))

        Label(search_row, text="Symbol:").pack(side="left")
        self.ticker_entry = Entry(search_row, width=18)
        self.ticker_entry.pack(side="left", padx=(8, 8))
        Button(search_row, text="Analyze", command=self.analyze_ticker).pack(side="left")

        Label(search_row, text="Account:").pack(side="left", padx=(18, 4))
        account_combo = Combobox(search_row, textvariable=self.ticker_account, values=["paper", "live"], state="readonly", width=10)
        account_combo.pack(side="left")

        self.graph_canvas = Canvas(frame, bg="#ffffff", height=260)
        self.graph_canvas.pack(fill="x", pady=(8, 8))

        self.ticker_status_label = Label(frame, text="Quote delay: unknown", font=("Segoe UI", 10, "italic"))
        self.ticker_status_label.pack(anchor="w", pady=(0, 4))

        self.ticker_result = ScrolledText(frame, height=10, wrap="word")
        self.ticker_result.pack(fill="x", pady=(0, 8))
        self.ticker_result.configure(state="disabled")

        model_frame = Frame(frame)
        model_frame.pack(fill="x", pady=(0, 8))
        Label(model_frame, text="Backtest models", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        for name in self.AVAILABLE_MODELS:
            Checkbutton(model_frame, text=name, variable=self.model_vars[name]).pack(anchor="w")

        run_row = Frame(frame)
        run_row.pack(fill="x", pady=(0, 8))
        Button(run_row, text="Run backtest", command=self.run_backtest).pack(side="left")

        controls = Frame(frame)
        controls.pack(fill="x", pady=(0, 8))

        Label(controls, text="Allocation amount:").pack(side="left")
        self.ticker_allocation = Entry(controls, width=14)
        self.ticker_allocation.insert(0, "250")
        self.ticker_allocation.pack(side="left", padx=(8, 8))

        Button(controls, text="Add this ticker", command=self.add_ticker_position).pack(side="left")

    def _build_trade_tab(self) -> None:
        frame = Frame(self.trade_tab, padding=12)
        frame.pack(fill="both", expand=True)

        Label(frame, text="Trade actions", font=("Segoe UI", 12, "bold")).pack(anchor="w")

        action_row = Frame(frame)
        action_row.pack(fill="x", pady=(8, 8))

        Button(action_row, text="Run S&P scan", command=self.run_sp500_scan).pack(side="left", padx=(0, 8))
        Button(action_row, text="Run profit target scan", command=self.run_profit_target_scan).pack(side="left", padx=(0, 8))
        Button(action_row, text="Run selected model scan", command=self.run_custom_scan).pack(side="left")

        account_row = Frame(frame)
        account_row.pack(fill="x", pady=(6, 8))

        Label(account_row, text="Account:").pack(side="left")
        self.trade_account = StringVar(value="paper")
        account_combo = Combobox(account_row, textvariable=self.trade_account, values=["paper", "live"], state="readonly", width=10)
        account_combo.pack(side="left", padx=(8, 8))

        self.pending_list = ScrolledText(frame, height=10, wrap="word")
        self.pending_list.pack(fill="x", pady=(6, 6))
        self.pending_list.configure(state="disabled")

        action_row_2 = Frame(frame)
        action_row_2.pack(fill="x", pady=(6, 8))

        Button(action_row_2, text="Accept selected trade", command=self.accept_selected_trade).pack(side="left", padx=(0, 8))
        Button(action_row_2, text="Clear pending", command=self.clear_pending_trades).pack(side="left")

        Label(frame, text="Alerts log", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(10, 6))
        self.alert_log = ScrolledText(frame, height=12, wrap="word")
        self.alert_log.pack(fill="both", expand=True)
        self.alert_log.configure(state="disabled")

    def _build_algorithm_tab(self) -> None:
        frame = Frame(self.algorithm_tab, padding=12)
        frame.pack(fill="both", expand=True)

        Label(frame, text="Algorithm builder", font=("Segoe UI", 12, "bold")).pack(anchor="w")

        builder_row = Frame(frame)
        builder_row.pack(fill="x", pady=(8, 8))

        Label(builder_row, text="Ticker:").pack(side="left")
        self.algorithm_entry = Entry(builder_row, width=18)
        self.algorithm_entry.pack(side="left", padx=(8, 8))
        Button(builder_row, text="Analyze model", command=self.build_algorithm).pack(side="left")
        Button(builder_row, text="Accept model", command=self.accept_current_algorithm_model).pack(side="left", padx=(8, 0))

        self.algorithm_result = ScrolledText(frame, height=16, wrap="word")
        self.algorithm_result.pack(fill="both", expand=True)
        self.algorithm_result.configure(state="disabled")

        controls = Frame(frame)
        controls.pack(fill="x", pady=(10, 0))

        Label(controls, text="Allocation amount:").pack(side="left")
        self.algorithm_allocation = Entry(controls, width=14)
        self.algorithm_allocation.insert(0, "250")
        self.algorithm_allocation.pack(side="left", padx=(8, 8))

        Button(controls, text="Run selected model scan", command=self.run_custom_scan).pack(side="left")

    def append_log(self, message: str) -> None:
        self.alert_log.configure(state="normal")
        self.alert_log.insert("end", f"{message}\n")
        self.alert_log.see("end")
        self.alert_log.configure(state="disabled")

    def refresh_account(self) -> None:
        snapshot = self.agent.get_account_snapshot("paper")
        cash = float(snapshot.get("cash", 0.0))
        positions = snapshot.get("positions", {})

        self.cash_label.configure(text=f"Cash: ${cash:,.2f}")

        if positions:
            position_lines = [f"{symbol}: qty {info['quantity']:.4f} | avg ${info['avg_price']:.2f}" for symbol, info in positions.items()]
            self.positions_label.configure(text="Positions:\n" + "\n".join(position_lines))
        else:
            self.positions_label.configure(text="Positions: none")

        history = self.agent.get_trade_history("paper")
        history_lines = [f"{record.timestamp} | {record.symbol} | {record.action} | {record.status} | qty {record.quantity:.4f}" for record in history]

        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", "end")
        if history_lines:
            self.history_text.insert("1.0", "\n".join(history_lines))
        else:
            self.history_text.insert("1.0", "No trades recorded yet.")
        self.history_text.configure(state="disabled")

    def _set_result_text(self, widget, message: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", message)
        widget.configure(state="disabled")

    def _selected_models(self) -> list[str]:
        return [name for name in self.AVAILABLE_MODELS if self.model_vars[name].get()]

    def _format_backtest_summary(self, results) -> str:
        lines = [f"Symbol: {self.current_symbol}", f"Initial cash: $1000.00", ""]
        for result in results:
            lines.extend(
                [
                    f"{result['model_name']}:",
                    f"  Trades: {result['trade_count']}",
                    f"  Buy events: {result['buy_count']}",
                    f"  Sell events: {result['sell_count']}",
                    f"  Final cash: ${result['final_cash']:,.2f}",
                    f"  Change: ${result['account_change']:,.2f}",
                    f"  Win rate: {result['win_rate']:.1%}",
                    f"  Avg return: {result['avg_return']:.1%}",
                    "",
                ]
            )
        if self.best_model_name:
            lines.append(f"Best model: {self.best_model_name}")
        return "\n".join(lines)

    def _render_price_chart(self, history: pd.DataFrame) -> None:
        self.graph_canvas.delete("all")
        if history.empty:
            self.graph_canvas.create_text(200, 120, text="No historical data available.", anchor="center")
            return

        self.graph_canvas.update_idletasks()
        width = max(self.graph_canvas.winfo_width(), 320)
        height = max(self.graph_canvas.winfo_height(), 220)
        padding = 40
        closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
        if closes.empty:
            self.graph_canvas.create_text(width // 2, height // 2, text="No close values available.", anchor="center")
            return

        min_price = float(closes.min())
        max_price = float(closes.max())
        price_span = max(max_price - min_price, 1.0)

        dates = list(history["Date"])
        x_step = (width - (padding * 2)) / max(len(dates) - 1, 1)
        points = []
        for index, price in enumerate(closes):
            x = padding + (index * x_step)
            y = height - padding - ((float(price) - min_price) / price_span) * (height - (padding * 2))
            points.append((x, y))

        for i in range(len(points) - 1):
            self.graph_canvas.create_line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], fill="#1f77b4", width=2)

        self.graph_canvas.create_line(padding, height - padding, width - padding, height - padding, fill="#222222", width=1)
        self.graph_canvas.create_line(padding, padding, padding, height - padding, fill="#222222", width=1)

        for label_index in range(0, len(dates), max(1, len(dates) // 5)):
            x = padding + (label_index * x_step)
            label = str(dates[label_index].strftime("%Y-%m-%d")) if hasattr(dates[label_index], "strftime") else str(dates[label_index])
            self.graph_canvas.create_text(x, height - 10, text=label, anchor="n", angle=30)

    def analyze_ticker(self) -> None:
        symbol = self.ticker_entry.get().strip().upper()
        if not symbol:
            messagebox.showwarning("Ticker Search", "Enter a ticker symbol.")
            return

        try:
            self.current_symbol = symbol
            data = self.agent.fetch_market_data(symbol, period="1y")
            self.last_history = data
            self._render_price_chart(data)

            quote = self.agent.fetch_latest_quote(symbol)
            self.ticker_status_label.configure(
                text=f"Latest quote: ${quote.price:.2f} at {quote.timestamp.isoformat()} (delay {quote.delay_seconds:.1f}s, source {quote.source})"
            )

            suggestion = self.agent.analyze_ticker_for_model(symbol, historical_data=data)
            self.current_suggestion = suggestion
            self.best_model_name = suggestion.model_name
            self._set_result_text(
                self.ticker_result,
                "\n".join(
                    [
                        f"Symbol: {suggestion.symbol}",
                        f"Latest quote: ${quote.price:.2f}",
                        f"Quote timestamp: {quote.timestamp.isoformat()}",
                        f"Delay: {quote.delay_seconds:.1f} seconds",
                        f"Current best model: {suggestion.model_name}",
                        f"Score: {suggestion.score:.3f}",
                        f"Confidence: {suggestion.confidence:.3f}",
                        f"Rationale: {suggestion.rationale}",
                    ]
                ),
            )
            self.append_log(f"Analyzed {symbol}: {suggestion.model_name} with {quote.delay_seconds:.1f}s quote delay")
        except Exception as exc:
            messagebox.showerror("Ticker Search", str(exc))

    def run_backtest(self) -> None:
        symbol = self.current_symbol or self.ticker_entry.get().strip().upper()
        if not symbol:
            messagebox.showwarning("Ticker Search", "Enter a ticker symbol before running the backtest.")
            return

        if self.last_history is None:
            self.analyze_ticker()
            if self.last_history is None:
                return

        selected = self._selected_models()
        if not selected:
            messagebox.showwarning("Ticker Search", "Choose at least one model to backtest.")
            return

        try:
            results = self.agent.backtest_ticker_models(
                symbol,
                historical_data=self.last_history,
                selected_models=selected,
                initial_cash=1000.0,
            )
            self.last_backtest = results
            self.best_model_name = max(results, key=lambda result: result["final_cash"])["model_name"]
            self._set_result_text(self.ticker_result, self._format_backtest_summary(results))
            self.append_log(f"Backtested {len(results)} model(s) for {symbol}.")
        except Exception as exc:
            messagebox.showerror("Ticker Search", str(exc))

    def add_ticker_position(self) -> None:
        symbol = self.current_symbol or self.ticker_entry.get().strip().upper()
        if not symbol:
            messagebox.showwarning("Ticker Search", "Enter a ticker symbol before adding it.")
            return

        if self.last_history is None:
            self.analyze_ticker()
            if self.last_history is None:
                return

        if not self.last_backtest:
            self.run_backtest()
            if not self.last_backtest:
                return

        try:
            allocation = float(self.ticker_allocation.get().strip())
            if allocation <= 0:
                raise ValueError("Allocation must be greater than zero.")

            model_name = self.best_model_name or (self.current_suggestion.model_name if self.current_suggestion else self.AVAILABLE_MODELS[0])
            account = self.ticker_account.get().strip()
            config = self.agent.add_ticker_position(
                symbol,
                model_name=model_name,
                allocation_amount=allocation,
                account=account,
                historical_data=self.last_history,
            )
            self._set_result_text(
                self.ticker_result,
                self.ticker_result.get("1.0", "end").strip()
                + f"\n\nAdded {symbol} to {account} using {model_name} at ${allocation:,.2f}."
            )
            self.append_log(f"Added {symbol} to {account} with {model_name} at ${allocation:,.2f}.")
            self.refresh_account()
        except Exception as exc:
            messagebox.showerror("Ticker Search", str(exc))

    def accept_current_ticker_model(self) -> None:
        if self.current_suggestion is None:
            messagebox.showwarning("Ticker Search", "Analyze a ticker first.")
            return

        try:
            allocation = float(self.ticker_allocation.get().strip())
            if allocation <= 0:
                raise ValueError("Allocation must be greater than zero.")
            config = self.agent.accept_custom_model(self.current_symbol, self.current_suggestion, allocation, account="paper")
            self.append_log(f"Accepted {config.model_name} for {config.symbol} at ${config.allocation_amount:,.2f}")
            self._set_result_text(self.ticker_result, self.ticker_result.get("1.0", "end").strip() + "\n\nAccepted custom model for paper account.")
        except Exception as exc:
            messagebox.showerror("Ticker Search", str(exc))

    def build_algorithm(self) -> None:
        symbol = self.algorithm_entry.get().strip().upper()
        if not symbol:
            messagebox.showwarning("Algorithm Builder", "Enter a ticker symbol.")
            return

        try:
            suggestion = self.agent.analyze_ticker_for_model(symbol)
            self.current_suggestion = suggestion
            self.current_symbol = symbol
            self._set_result_text(
                self.algorithm_result,
                "\n".join(
                    [
                        f"Symbol: {suggestion.symbol}",
                        f"Suggested model: {suggestion.model_name}",
                        f"Score: {suggestion.score:.3f}",
                        f"Confidence: {suggestion.confidence:.3f}",
                        f"Rationale: {suggestion.rationale}",
                    ]
                ),
            )
            self.append_log(f"Built algorithm suggestion for {symbol}: {suggestion.model_name}")
        except Exception as exc:
            messagebox.showerror("Algorithm Builder", str(exc))

    def accept_current_algorithm_model(self) -> None:
        if self.current_suggestion is None:
            messagebox.showwarning("Algorithm Builder", "Analyze a ticker first.")
            return

        try:
            allocation = float(self.algorithm_allocation.get().strip())
            if allocation <= 0:
                raise ValueError("Allocation must be greater than zero.")
            config = self.agent.accept_custom_model(self.current_symbol, self.current_suggestion, allocation, account="paper")
            self.append_log(f"Accepted algorithm {config.model_name} for {config.symbol} at ${config.allocation_amount:,.2f}")
            self._set_result_text(self.algorithm_result, self.algorithm_result.get("1.0", "end").strip() + "\n\nAccepted custom model for paper account.")
        except Exception as exc:
            messagebox.showerror("Algorithm Builder", str(exc))

    def _store_pending_signals(self, signals) -> None:
        self.pending_signals = list(signals)
        self.pending_list.configure(state="normal")
        self.pending_list.delete("1.0", "end")
        if not signals:
            self.pending_list.insert("1.0", "No pending trades.")
        else:
            lines = [f"{signal.model_name} | {signal.symbol} | {signal.action} | qty {signal.quantity:.4f} | price ${signal.price:.2f}" for signal in signals]
            self.pending_list.insert("1.0", "\n".join(lines))
        self.pending_list.configure(state="disabled")

    def run_sp500_scan(self) -> None:
        try:
            signals = self.agent.run_sp500_scan()
            self._store_pending_signals(signals)
            self.append_log(f"S&P scan returned {len(signals)} signal(s).")
        except Exception as exc:
            messagebox.showerror("Trade", str(exc))

    def run_profit_target_scan(self) -> None:
        try:
            signals = self.agent.run_profit_target_scan(account="paper")
            self._store_pending_signals(signals)
            self.append_log(f"Profit target scan returned {len(signals)} signal(s).")
        except Exception as exc:
            messagebox.showerror("Trade", str(exc))

    def run_custom_scan(self) -> None:
        try:
            symbol = self.current_symbol or self.ticker_entry.get().strip().upper()
            if not symbol:
                messagebox.showwarning("Trade", "Select or analyze a ticker before running the scan.")
                return

            if symbol in self.agent.custom_models:
                model_name = self.agent.custom_models[symbol].model_name
            elif self.best_model_name:
                model_name = self.best_model_name
            elif self.current_suggestion is not None:
                model_name = self.current_suggestion.model_name
            else:
                model_name = None

            if model_name is None:
                messagebox.showwarning("Trade", "Accept a custom model or analyze a ticker before running the scan.")
                return

            allocation = float(self.ticker_allocation.get().strip()) if self.ticker_allocation.get().strip() else 250.0
            signals = self.agent.scan_model(symbol, model_name, account="paper", allocation_amount=allocation, historical_data=self.last_history)
            self._store_pending_signals(signals)
            self.append_log(f"Selected model scan returned {len(signals)} signal(s) for {symbol} using {model_name}.")
        except Exception as exc:
            messagebox.showerror("Trade", str(exc))

    def accept_selected_trade(self) -> None:
        if not self.pending_signals:
            messagebox.showwarning("Trade", "No pending trades to accept.")
            return

        try:
            account = self.trade_account.get().strip()
            selected = self.pending_signals[0]
            record = self.agent.accept_trade(selected, account=account)
            self.append_log(f"Accepted trade {record.symbol} for {account}: {record.action} ({record.status})")
            self.refresh_account()
            self.clear_pending_trades()
        except Exception as exc:
            messagebox.showerror("Trade", str(exc))

    def clear_pending_trades(self) -> None:
        self.pending_signals = []
        self.pending_list.configure(state="normal")
        self.pending_list.delete("1.0", "end")
        self.pending_list.insert("1.0", "No pending trades.")
        self.pending_list.configure(state="disabled")


def launch_trading_gui(root: Tk | None = None) -> Tk:
    app = TradingBotWindow(root=root)
    app.root.mainloop()
    return app.root
