from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
import json
import smtplib
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yfinance as yf


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class Signal:
    model_name: str
    symbol: str
    action: str
    reason: str
    confidence: float
    price: float
    quantity: float
    account: str
    timestamp: str = field(default_factory=_utc_timestamp)


@dataclass
class TradeRecord:
    trade_id: str
    account: str
    model_name: str
    symbol: str
    action: str
    quantity: float
    price: float
    status: str
    timestamp: str
    notes: str = ""


@dataclass
class ModelSuggestion:
    symbol: str
    model_name: str
    score: float
    confidence: float
    rationale: str
    parameters: Dict[str, float]


@dataclass
class CustomModelConfig:
    symbol: str
    model_name: str
    account: str
    allocation_amount: float
    parameters: Dict[str, float]


@dataclass
class MarketQuote:
    symbol: str
    price: float
    timestamp: datetime
    delay_seconds: float
    source: str


class AlertManager:
    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        sender_email: Optional[str] = None,
        recipient_email: Optional[str] = None,
        use_email: bool = False,
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.sender_email = sender_email
        self.recipient_email = recipient_email
        self.use_email = use_email

    def send(self, title: str, message: str) -> None:
        print(f"[ALERT] {title}: {message}")
        if not self.use_email:
            return
        if not all([self.smtp_server, self.smtp_user, self.smtp_password, self.sender_email, self.recipient_email]):
            raise ValueError("Email alerting requires smtp_server, smtp_user, smtp_password, sender_email, and recipient_email.")

        msg = EmailMessage()
        msg["Subject"] = title
        msg["From"] = self.sender_email
        msg["To"] = self.recipient_email
        msg.set_content(message)

        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)


class TradeLedger:
    def __init__(self, ledger_path: str | Path):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: List[TradeRecord] = self._load()

    def _load(self) -> List[TradeRecord]:
        if not self.ledger_path.exists():
            return []

        payload = json.loads(self.ledger_path.read_text())
        return [TradeRecord(**item) for item in payload]

    def save(self) -> None:
        payload = [
            {
                "trade_id": record.trade_id,
                "account": record.account,
                "model_name": record.model_name,
                "symbol": record.symbol,
                "action": record.action,
                "quantity": record.quantity,
                "price": record.price,
                "status": record.status,
                "timestamp": record.timestamp,
                "notes": record.notes,
            }
            for record in self._records
        ]
        self.ledger_path.write_text(json.dumps(payload, indent=2))

    def add_record(self, record: TradeRecord) -> None:
        self._records.append(record)
        self.save()

    def get_records(self, account: Optional[str] = None) -> List[TradeRecord]:
        if account is None:
            return list(self._records)
        return [record for record in self._records if record.account == account]

    def update_status(self, trade_id: str, status: str, notes: str = "") -> None:
        for record in self._records:
            if record.trade_id == trade_id:
                record.status = status
                if notes:
                    record.notes = notes
                self.save()
                return


class PortfolioManager:
    def __init__(self, account_name: str, starting_cash: float, ledger: TradeLedger):
        self.account_name = account_name
        self.starting_cash = starting_cash
        self.ledger = ledger
        self.cash = starting_cash
        self.positions: Dict[str, Dict[str, float]] = {}
        self._load_state()

    def _load_state(self) -> None:
        self.cash = self.starting_cash
        self.positions = {}
        for trade in self.ledger.get_records(self.account_name):
            self._apply_trade_record(trade, persist=False)

    def _apply_trade_record(self, trade: TradeRecord, persist: bool = True) -> None:
        if trade.action == "deposit":
            self.cash += trade.quantity
        elif trade.action == "withdraw":
            self.cash -= trade.quantity
        elif trade.action in {"buy", "long"}:
            current = self.positions.get(trade.symbol, {"quantity": 0.0, "avg_price": 0.0})
            current["quantity"] += trade.quantity
            current["avg_price"] = ((current["avg_price"] * (current["quantity"] - trade.quantity)) + trade.price * trade.quantity) / current["quantity"]
            self.positions[trade.symbol] = current
            self.cash -= trade.quantity * trade.price
        elif trade.action in {"sell", "short"}:
            if trade.symbol not in self.positions:
                raise ValueError(f"Cannot sell {trade.symbol}; position is not open.")
            current = self.positions[trade.symbol]
            if current["quantity"] < trade.quantity:
                raise ValueError(f"Cannot sell {trade.quantity} shares of {trade.symbol}; only {current['quantity']} are held.")
            current["quantity"] -= trade.quantity
            self.cash += trade.quantity * trade.price
            if current["quantity"] == 0:
                del self.positions[trade.symbol]

        if persist:
            self.ledger.save()

    def add_cash(self, amount: float, notes: str = "") -> TradeRecord:
        if amount <= 0:
            raise ValueError("amount must be greater than 0")

        record = TradeRecord(
            trade_id=f"{self.account_name}-BANK-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            account=self.account_name,
            model_name="BankTransaction",
            symbol="BANK",
            action="deposit",
            quantity=amount,
            price=1.0,
            status="accepted",
            timestamp=_utc_timestamp(),
            notes=notes or "Bank transfer added to account.",
        )
        self._apply_trade_record(record, persist=False)
        self.ledger.add_record(record)
        return record

    def accept_order(self, signal: Signal) -> TradeRecord:
        if signal.action in {"buy", "long"}:
            cost = signal.quantity * signal.price
            if self.cash < cost:
                raise ValueError(f"Insufficient cash in {self.account_name}: needed ${cost:.2f}, available ${self.cash:.2f}")
        elif signal.action in {"sell", "short"}:
            if signal.symbol not in self.positions:
                raise ValueError(f"Cannot sell {signal.symbol}; position is not open.")
            current = self.positions[signal.symbol]
            if current["quantity"] < signal.quantity:
                raise ValueError(f"Cannot sell {signal.quantity} shares of {signal.symbol}; only {current['quantity']} are held.")

        record = TradeRecord(
            trade_id=f"{self.account_name}-{signal.symbol}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            account=self.account_name,
            model_name=signal.model_name,
            symbol=signal.symbol,
            action=signal.action,
            quantity=signal.quantity,
            price=signal.price,
            status="accepted",
            timestamp=_utc_timestamp(),
            notes=f"Accepted via {signal.model_name}",
        )
        self._apply_trade_record(record, persist=False)
        self.ledger.add_record(record)
        return record

    def get_snapshot(self) -> Dict[str, float | Dict[str, Dict[str, float]]]:
        return {"cash": self.cash, "positions": self.positions}


class BaseModel:
    def evaluate(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        raise NotImplementedError


class SP500DropModel(BaseModel):
    def __init__(self, buy_amount: float = 500.0, drop_threshold: float = 2.5):
        self.buy_amount = buy_amount
        self.drop_threshold = drop_threshold

    def evaluate(self, data: pd.DataFrame, symbol: str = "^GSPC") -> List[Signal]:
        if data.empty:
            return []

        dt = data.copy()
        if "Daily_Change_Percent" not in dt.columns:
            dt["Daily_Change_Percent"] = dt["Close"].pct_change() * 100

        latest = dt.iloc[-1]
        current_price = float(latest["Close"])
        daily_change = float(latest["Daily_Change_Percent"])

        if daily_change <= -self.drop_threshold:
            quantity = self.buy_amount / current_price
            return [
                Signal(
                    model_name="SP500DropModel",
                    symbol=symbol,
                    action="buy",
                    reason=f"S&P 500 dropped {daily_change:.2f}% today; buying ${self.buy_amount:.2f} into the index.",
                    confidence=min(1.0, abs(daily_change) / 10.0),
                    price=current_price,
                    quantity=quantity,
                    account="paper",
                )
            ]
        return []


class TrendFollowingModel(BaseModel):
    def __init__(self, fast_window: int = 5, slow_window: int = 20, entry_threshold: float = 0.02, exit_threshold: float = -0.01):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def evaluate(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        if data.empty or len(data) < max(self.fast_window, self.slow_window):
            return []

        dt = data.copy()
        dt["Close"] = pd.to_numeric(dt["Close"], errors="coerce")
        dt = dt.dropna(subset=["Close"])
        if dt.empty:
            return []

        close = dt["Close"].astype(float)
        fast_ma = close.rolling(self.fast_window).mean().iloc[-1]
        slow_ma = close.rolling(self.slow_window).mean().iloc[-1]
        current_price = float(close.iloc[-1])

        if pd.isna(fast_ma) or pd.isna(slow_ma):
            return []

        trend_pct = (fast_ma - slow_ma) / slow_ma
        if trend_pct >= self.entry_threshold:
            quantity = 100.0 / current_price
            return [
                Signal(
                    model_name="TrendFollowingModel",
                    symbol=symbol,
                    action="buy",
                    reason=f"Fast MA is {trend_pct * 100:.2f}% above slow MA; trend is positive.",
                    confidence=min(1.0, trend_pct / 0.05),
                    price=current_price,
                    quantity=quantity,
                    account="paper",
                )
            ]

        if trend_pct <= self.exit_threshold:
            return [
                Signal(
                    model_name="TrendFollowingModel",
                    symbol=symbol,
                    action="sell",
                    reason=f"Fast MA is {trend_pct * 100:.2f}% below slow MA; trend is negative.",
                    confidence=min(1.0, abs(trend_pct) / 0.05),
                    price=current_price,
                    quantity=0.0,
                    account="paper",
                )
            ]

        return []


class MultiAssetDiversificationModel(BaseModel):
    def __init__(self, target_weights: Optional[Dict[str, float]] = None):
        self.target_weights = target_weights or {
            "SPY": 0.35,
            "BND": 0.25,
            "BTC-USD": 0.15,
            "EURUSD=X": 0.15,
            "GC=F": 0.10,
        }

    def evaluate(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        if symbol not in self.target_weights:
            return []

        price = float(data["Close"].iloc[-1])
        allocation = self.target_weights[symbol]
        quantity = (1000.0 * allocation) / price

        return [
            Signal(
                model_name="MultiAssetDiversificationModel",
                symbol=symbol,
                action="buy",
                reason=f"Diversification model recommends a {allocation * 100:.0f}% weight for {symbol}.",
                confidence=0.6,
                price=price,
                quantity=quantity,
                account="paper",
            )
        ]


class DividendCaptureModel(BaseModel):
    def __init__(self, dividend_window: int = 45, min_yield: float = 0.01):
        self.dividend_window = dividend_window
        self.min_yield = min_yield

    def _normalize_dividend_data(self, dividend_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        if dividend_data is None:
            return pd.DataFrame(columns=["Date", "Dividends"])

        df = dividend_data.copy()
        if df.empty:
            return pd.DataFrame(columns=["Date", "Dividends"])

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        elif "index" in df.columns:
            df["Date"] = pd.to_datetime(df["index"], errors="coerce")
        else:
            return pd.DataFrame(columns=["Date", "Dividends"])

        df = df.dropna(subset=["Date"])
        if "Dividends" not in df.columns:
            if "dividends" in df.columns:
                df = df.rename(columns={"dividends": "Dividends"})
            else:
                return pd.DataFrame(columns=["Date", "Dividends"])

        df["Dividends"] = pd.to_numeric(df["Dividends"], errors="coerce")
        return df.dropna(subset=["Dividends"]).reset_index(drop=True)

    def _current_price(self, history: pd.DataFrame) -> float:
        if history.empty or "Close" not in history.columns:
            return 0.0
        close = pd.to_numeric(history["Close"], errors="coerce")
        close = close.dropna()
        if close.empty:
            return 0.0
        return float(close.iloc[-1])

    def _recent_return(self, history: pd.DataFrame) -> float:
        if history.empty or "Close" not in history.columns:
            return 0.0
        close = pd.to_numeric(history["Close"], errors="coerce").dropna()
        if len(close) < 2:
            return 0.0
        return float((close.iloc[-1] / close.iloc[-2]) - 1.0)

    def _next_dividend(self, dividend_data: pd.DataFrame, current_date: pd.Timestamp) -> Optional[dict]:
        events = self._normalize_dividend_data(dividend_data)
        if events.empty:
            return None

        events = events.sort_values("Date").reset_index(drop=True)
        future_events = events[events["Date"] >= current_date]
        if not future_events.empty:
            chosen = future_events.iloc[0]
        else:
            chosen = events.iloc[-1]

        return {
            "date": pd.Timestamp(chosen["Date"]),
            "amount": float(chosen["Dividends"]),
        }

    def analyze_symbol(self, symbol: str, history: pd.DataFrame, dividend_data: Optional[pd.DataFrame] = None) -> Optional[Dict[str, object]]:
        if history.empty:
            return None

        current_price = self._current_price(history)
        if current_price <= 0:
            return None

        current_date = pd.Timestamp.now("UTC")
        next_dividend = self._next_dividend(dividend_data, current_date)
        if next_dividend is None:
            return None

        days_to_dividend = max(int((next_dividend["date"] - current_date).days), 0)
        yield_pct = next_dividend["amount"] / current_price
        recent_return = self._recent_return(history)
        hold_is_better = recent_return >= yield_pct

        score = max(0.0, yield_pct * (1 - min(days_to_dividend / max(self.dividend_window, 1), 1.0)))
        events = self._normalize_dividend_data(dividend_data)
        total_dividends = float(events["Dividends"].sum()) if not events.empty else 0.0
        dividend_count = int(len(events))

        eligible = days_to_dividend <= self.dividend_window and yield_pct >= self.min_yield
        return {
            "symbol": symbol,
            "current_price": current_price,
            "next_dividend_amount": next_dividend["amount"],
            "next_dividend_date": next_dividend["date"],
            "days_to_dividend": days_to_dividend,
            "yield_pct": yield_pct,
            "recent_return": recent_return,
            "hold_is_better": hold_is_better,
            "recommended_action": "hold" if hold_is_better else "rotate",
            "score": score,
            "dividend_count": dividend_count,
            "total_dividends": total_dividends,
            "eligible": eligible,
        }

    def evaluate(self, data: pd.DataFrame, symbol: str) -> List[Signal]:
        dividend_data = pd.DataFrame(columns=["Date", "Dividends"])
        if "Dividends" in data.columns:
            dividend_data = data[["Date", "Dividends"]].copy()

        summary = self.analyze_symbol(symbol, data, dividend_data)
        if summary is None:
            return []

        if summary["days_to_dividend"] > self.dividend_window or summary["yield_pct"] < self.min_yield:
            return []

        current_price = float(summary["current_price"])
        quantity = 1000.0 / current_price
        return [
            Signal(
                model_name="DividendCaptureModel",
                symbol=symbol,
                action="buy",
                reason=f"Dividend capture model sees {symbol} paying a {summary['yield_pct'] * 100:.2f}% yield with the next dividend in {summary['days_to_dividend']} days.",
                confidence=min(1.0, summary["yield_pct"] / 0.05),
                price=current_price,
                quantity=quantity,
                account="paper",
            )
        ]


class TradingAgent:
    def __init__(
        self,
        data_dir: str | Path = "data",
        alert_manager: Optional[AlertManager] = None,
        paper_starting_cash: float = 10000.0,
        live_starting_cash: float = 10000.0,
        paper_ledger_path: str | Path = "paper_trades.json",
        live_ledger_path: str | Path = "live_trades.json",
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.alert_manager = alert_manager or AlertManager()

        self.paper_ledger = TradeLedger(self.data_dir / paper_ledger_path)
        self.live_ledger = TradeLedger(self.data_dir / live_ledger_path)

        self.paper_account = PortfolioManager("paper", paper_starting_cash, self.paper_ledger)
        self.live_account = PortfolioManager("live", live_starting_cash, self.live_ledger)

        self.models = {
            "sp500": SP500DropModel(),
            "trend": TrendFollowingModel(),
            "diversification": MultiAssetDiversificationModel(),
            "dividend": DividendCaptureModel(),
        }
        self.custom_models: Dict[str, CustomModelConfig] = {}

    def load_csv(self, csv_path: str | Path) -> pd.DataFrame:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        df = pd.read_csv(path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        return df

    def fetch_market_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d", actions=False, auto_adjust=False, progress=False)
        if df.empty:
            raise ValueError(f"No market data returned for {symbol}")
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        if "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "Adj_Close"})
        if "Close" not in df.columns:
            raise ValueError(f"Missing Close column for {symbol}")
        df["Daily_Change_Percent"] = df["Close"].pct_change() * 100
        return df

    def fetch_latest_quote(self, symbol: str) -> MarketQuote:
        ticker = yf.Ticker(symbol)
        price = None
        timestamp = None
        source = "yfinance.fast_info"

        fast_info = getattr(ticker, "fast_info", {}) or {}
        if isinstance(fast_info, dict):
            price = fast_info.get("last_price") or fast_info.get("regular_market_price")
            last_trade = fast_info.get("last_trade_time") or fast_info.get("regular_market_time") or fast_info.get("last_trade_timestamp")
            if last_trade is not None:
                if isinstance(last_trade, (int, float)):
                    timestamp = datetime.fromtimestamp(float(last_trade), tz=timezone.utc)
                elif isinstance(last_trade, datetime):
                    timestamp = last_trade.astimezone(timezone.utc)

        if price is None or timestamp is None:
            info = getattr(ticker, "info", {}) or {}
            if isinstance(info, dict):
                price = price or info.get("regularMarketPrice") or info.get("previousClose")
                last_trade = info.get("regularMarketTime") or info.get("postMarketTime") or info.get("preMarketTime")
                if last_trade is not None and isinstance(last_trade, (int, float)):
                    timestamp = datetime.fromtimestamp(float(last_trade), tz=timezone.utc)

        if price is None or timestamp is None:
            source = "yfinance.history"
            history = ticker.history(period="1d", interval="1m", actions=False, progress=False)
            if history.empty:
                raise ValueError(f"Unable to retrieve live quote for {symbol}")
            recent = history.reset_index().iloc[-1]
            price = float(recent["Close"])
            timestamp = recent["Datetime"]
            if isinstance(timestamp, pd.Timestamp):
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize(timezone.utc)
                else:
                    timestamp = timestamp.tz_convert(timezone.utc)

        if price is None or timestamp is None:
            raise ValueError(f"Unable to retrieve live quote for {symbol}")

        delay = max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds())
        return MarketQuote(symbol=symbol, price=float(price), timestamp=timestamp, delay_seconds=delay, source=source)

    def _prepare_history(self, data: pd.DataFrame) -> pd.DataFrame:
        dt = data.copy()
        if dt.empty:
            return dt
        dt["Close"] = pd.to_numeric(dt["Close"], errors="coerce")
        dt = dt.dropna(subset=["Close"]).reset_index(drop=True)
        if dt.empty:
            return dt
        dt["SMA5"] = dt["Close"].rolling(5).mean()
        dt["SMA20"] = dt["Close"].rolling(20).mean()
        dt["SMA50"] = dt["Close"].rolling(50).mean()
        dt["ROC5"] = dt["Close"].pct_change(5)
        dt["ROC10"] = dt["Close"].pct_change(10)
        dt["ROC20"] = dt["Close"].pct_change(20)
        dt["High20"] = dt["Close"].rolling(20).max()
        return dt

    def _rule_moving_average_cross(self, data: pd.DataFrame, idx: int) -> bool:
        if idx < 1:
            return False
        close = float(data["Close"].iloc[idx])
        sma5 = data["SMA5"].iloc[idx]
        sma20 = data["SMA20"].iloc[idx]
        prev_sma5 = data["SMA5"].iloc[idx - 1]
        prev_sma20 = data["SMA20"].iloc[idx - 1]
        return bool(pd.notna(sma5) and pd.notna(sma20) and pd.notna(prev_sma5) and pd.notna(prev_sma20) and close > sma20 and sma5 > sma20 and prev_sma5 <= prev_sma20)

    def _rule_momentum_breakout(self, data: pd.DataFrame, idx: int) -> bool:
        if idx < 20:
            return False
        close = float(data["Close"].iloc[idx])
        roc5 = data["ROC5"].iloc[idx]
        high20 = data["High20"].iloc[idx]
        return bool(pd.notna(roc5) and pd.notna(high20) and roc5 >= 0.03 and close >= high20 * 0.98)

    def _rule_pullback_after_momentum(self, data: pd.DataFrame, idx: int, pullback_pct: float = 0.02, momentum_return: float = 0.08, trend_return: float = 0.12) -> bool:
        if idx < 20:
            return False
        close = float(data["Close"].iloc[idx])
        sma20 = data["SMA20"].iloc[idx]
        roc10 = data["ROC10"].iloc[idx]
        roc20 = data["ROC20"].iloc[idx]
        return bool(
            pd.notna(sma20)
            and pd.notna(roc10)
            and pd.notna(roc20)
            and close >= sma20 * (1 - pullback_pct)
            and close > sma20
            and roc10 >= momentum_return
            and roc20 >= trend_return
        )

    def _simulate_rule(self, data: pd.DataFrame, rule_fn, target_return: float = 0.10, max_holding_days: int = 60) -> List[float]:
        trades: List[float] = []
        open_idx = None
        open_price = None

        for idx in range(20, len(data)):
            if open_idx is None and rule_fn(data, idx):
                open_idx = idx
                open_price = float(data["Close"].iloc[idx])
                continue

            if open_idx is not None:
                current_price = float(data["Close"].iloc[idx])
                if current_price >= open_price * (1 + target_return):
                    trades.append((current_price / open_price) - 1.0)
                    open_idx = None
                    open_price = None
                elif idx - open_idx >= max_holding_days:
                    trades.append((current_price / open_price) - 1.0)
                    open_idx = None
                    open_price = None

        if open_idx is not None:
            closing_price = float(data["Close"].iloc[-1])
            trades.append((closing_price / open_price) - 1.0)

        return trades

    def _score_rule(self, trades: List[float], target_return: float = 0.10) -> Dict[str, float]:
        if not trades:
            return {"score": 0.0, "success_rate": 0.0, "avg_return": 0.0, "trade_count": 0}

        success_count = sum(1 for trade in trades if trade >= target_return)
        avg_return = sum(trades) / len(trades)
        score = (success_count / len(trades)) * 0.65 + min(avg_return / target_return, 1.0) * 0.35
        return {"score": score, "success_rate": success_count / len(trades), "avg_return": avg_return, "trade_count": len(trades)}

    def _build_rule_suggestions(self, data: pd.DataFrame) -> List[Dict[str, object]]:
        summary: List[Dict[str, object]] = []

        moving_avg_trades = self._simulate_rule(data, self._rule_moving_average_cross)
        moving_avg_stats = self._score_rule(moving_avg_trades)
        summary.append(
            {
                "model_name": "MovingAverageCross",
                "score": moving_avg_stats["score"],
                "confidence": max(0.0, min(1.0, moving_avg_stats["success_rate"])),
                "rationale": f"This rule uses a 5-day over 20-day moving-average crossover and showed a {moving_avg_stats['success_rate']:.0%} success rate in backtesting.",
                "parameters": {"pullback_pct": 0.0, "momentum_return": 0.0, "trend_return": 0.0},
            }
        )

        breakout_trades = self._simulate_rule(data, self._rule_momentum_breakout)
        breakout_stats = self._score_rule(breakout_trades)
        summary.append(
            {
                "model_name": "MomentumBreakout",
                "score": breakout_stats["score"],
                "confidence": max(0.0, min(1.0, breakout_stats["success_rate"])),
                "rationale": f"This rule looks for a breakout above the 20-day high and a 5-day move of at least 3%, with a {breakout_stats['success_rate']:.0%} success rate.",
                "parameters": {"pullback_pct": 0.0, "momentum_return": 0.03, "trend_return": 0.0},
            }
        )

        pullback_trades = self._simulate_rule(data, self._rule_pullback_after_momentum)
        pullback_stats = self._score_rule(pullback_trades)
        summary.append(
            {
                "model_name": "PullbackAfterMomentum",
                "score": pullback_stats["score"],
                "confidence": max(0.0, min(1.0, pullback_stats["success_rate"])),
                "rationale": f"This rule buys after a strong rally when price pulls back toward the 20-day SMA, and showed a {pullback_stats['success_rate']:.0%} success rate.",
                "parameters": {"pullback_pct": 0.02, "momentum_return": 0.08, "trend_return": 0.12},
            }
        )

        return summary

    def analyze_ticker_for_model(self, symbol: str, period: str = "5y", historical_data: Optional[pd.DataFrame] = None, target_return: float = 0.10) -> ModelSuggestion:
        data = historical_data if historical_data is not None else self.fetch_market_data(symbol, period=period)
        prepared = self._prepare_history(data)
        if prepared.empty or len(prepared) < 40:
            raise ValueError(f"Not enough historical data to analyze {symbol}.")

        candidates = self._build_rule_suggestions(prepared)
        dividend_summary = self.models["dividend"].analyze_symbol(symbol, prepared, self._fetch_dividend_data(symbol))
        if dividend_summary is not None and dividend_summary["eligible"]:
            candidates.append(
                {
                    "model_name": "DividendCaptureModel",
                    "score": float(dividend_summary["score"]),
                    "confidence": min(1.0, float(dividend_summary["yield_pct"]) / 0.05),
                    "rationale": f"Dividend capture model sees a {dividend_summary['yield_pct'] * 100:.2f}% yield with the next dividend in {dividend_summary['days_to_dividend']} days.",
                    "parameters": {"dividend_window": self.models["dividend"].dividend_window, "min_yield": self.models["dividend"].min_yield},
                }
            )

        best = max(candidates, key=lambda item: item["score"])

        return ModelSuggestion(
            symbol=symbol,
            model_name=str(best["model_name"]),
            score=float(best["score"]),
            confidence=float(best["confidence"]),
            rationale=str(best["rationale"]),
            parameters=dict(best["parameters"]),
        )

    def _get_rule_fn(self, model_name: str):
        rule_map = {
            "MovingAverageCross": self._rule_moving_average_cross,
            "MomentumBreakout": self._rule_momentum_breakout,
            "PullbackAfterMomentum": self._rule_pullback_after_momentum,
        }
        if model_name not in rule_map:
            raise ValueError(f"Unsupported backtest model: {model_name}")
        return rule_map[model_name]

    def _simulate_dividend_capture_backtest(
        self,
        symbol: str,
        data: pd.DataFrame,
        dividend_data: Optional[pd.DataFrame],
        initial_cash: float = 1000.0,
        target_return: float = 0.10,
    ) -> Dict[str, object]:
        summary = self.models["dividend"].analyze_symbol(symbol, data, dividend_data)
        if summary is None:
            return {
                "model_name": "DividendCaptureModel",
                "trade_count": 0,
                "buy_count": 0,
                "sell_count": 0,
                "final_cash": initial_cash,
                "account_change": 0.0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "trade_events": [],
            }

        strategy_return = float(summary["recent_return"] if summary["hold_is_better"] else summary["yield_pct"])
        if pd.isna(strategy_return):
            strategy_return = 0.0

        final_cash = initial_cash * (1.0 + max(strategy_return, 0.0))
        return {
            "model_name": "DividendCaptureModel",
            "trade_count": 1 if summary["eligible"] else 0,
            "buy_count": 1 if summary["eligible"] else 0,
            "sell_count": 1 if summary["eligible"] and not summary["hold_is_better"] else 0,
            "final_cash": final_cash,
            "account_change": final_cash - initial_cash,
            "win_rate": 1.0 if strategy_return >= target_return else 0.0,
            "avg_return": strategy_return,
            "trade_events": [],
        }

    def _simulate_model_backtest(
        self,
        data: pd.DataFrame,
        model_name: str,
        initial_cash: float = 1000.0,
        target_return: float = 0.10,
        max_holding_days: int = 60,
    ) -> Dict[str, object]:
        prepared = self._prepare_history(data)
        if prepared.empty or len(prepared) < 40:
            return {
                "model_name": model_name,
                "trade_count": 0,
                "buy_count": 0,
                "sell_count": 0,
                "final_cash": initial_cash,
                "account_change": 0.0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "trade_events": [],
            }

        rule_fn = self._get_rule_fn(model_name)
        cash = float(initial_cash)
        position = None
        trade_events: List[Dict[str, float]] = []
        buy_count = 0
        sell_count = 0

        for idx in range(20, len(prepared)):
            current_price = float(prepared["Close"].iloc[idx])
            if position is None and rule_fn(prepared, idx):
                qty = cash / current_price
                position = {
                    "entry_idx": idx,
                    "entry_price": current_price,
                    "quantity": qty,
                }
                buy_count += 1
                continue

            if position is None:
                continue

            if current_price >= position["entry_price"] * (1 + target_return) or idx - position["entry_idx"] >= max_holding_days:
                cash += position["quantity"] * current_price
                exit_return = (current_price / position["entry_price"]) - 1.0
                trade_events.append(
                    {
                        "entry_idx": position["entry_idx"],
                        "exit_idx": idx,
                        "entry_price": position["entry_price"],
                        "exit_price": current_price,
                        "return": exit_return,
                    }
                )
                sell_count += 1
                position = None

        if position is not None:
            current_price = float(prepared["Close"].iloc[-1])
            cash += position["quantity"] * current_price
            exit_return = (current_price / position["entry_price"]) - 1.0
            trade_events.append(
                {
                    "entry_idx": position["entry_idx"],
                    "exit_idx": len(prepared) - 1,
                    "entry_price": position["entry_price"],
                    "exit_price": current_price,
                    "return": exit_return,
                }
            )
            sell_count += 1

        if trade_events:
            avg_return = sum(event["return"] for event in trade_events) / len(trade_events)
            win_rate = sum(1 for event in trade_events if event["return"] >= target_return) / len(trade_events)
        else:
            avg_return = 0.0
            win_rate = 0.0

        return {
            "model_name": model_name,
            "trade_count": len(trade_events),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "final_cash": cash,
            "account_change": cash - initial_cash,
            "win_rate": win_rate,
            "avg_return": avg_return,
            "trade_events": trade_events,
        }

    def backtest_ticker_models(
        self,
        symbol: str,
        historical_data: Optional[pd.DataFrame] = None,
        selected_models: Optional[List[str]] = None,
        initial_cash: float = 1000.0,
        target_return: float = 0.10,
        dividend_data_by_symbol: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> List[Dict[str, object]]:
        data = historical_data if historical_data is not None else self.fetch_market_data(symbol, period="1y")
        prepared = self._prepare_history(data)
        models = selected_models or ["MovingAverageCross", "MomentumBreakout", "PullbackAfterMomentum"]
        if prepared.empty:
            raise ValueError(f"Not enough historical data to backtest {symbol}.")
        if len(prepared) < 40 and "DividendCaptureModel" not in models:
            raise ValueError(f"Not enough historical data to backtest {symbol}.")
        dividend_data = None
        if dividend_data_by_symbol is not None:
            dividend_data = dividend_data_by_symbol.get(symbol)
        results = []
        for model_name in models:
            if model_name == "DividendCaptureModel":
                result = self._simulate_dividend_capture_backtest(
                    symbol,
                    prepared,
                    dividend_data,
                    initial_cash=initial_cash,
                    target_return=target_return,
                )
            else:
                result = self._simulate_model_backtest(
                    prepared,
                    model_name,
                    initial_cash=initial_cash,
                    target_return=target_return,
                )
            results.append(result)

        return results

    def add_ticker_position(
        self,
        symbol: str,
        model_name: str,
        allocation_amount: float,
        account: str = "paper",
        historical_data: Optional[pd.DataFrame] = None,
    ) -> CustomModelConfig:
        if allocation_amount <= 0:
            raise ValueError("allocation_amount must be greater than 0")
        if account not in {"paper", "live"}:
            raise ValueError("account must be 'paper' or 'live'")

        data = historical_data if historical_data is not None else self.fetch_market_data(symbol, period="1y")
        prepared = self._prepare_history(data)
        if prepared.empty:
            raise ValueError(f"Not enough historical data to add {symbol}.")

        current_price = float(prepared["Close"].iloc[-1])
        quantity = allocation_amount / current_price
        signal = Signal(
            model_name=model_name,
            symbol=symbol,
            action="buy",
            reason=f"Added {symbol} with algorithm {model_name} at ${allocation_amount:,.2f}.",
            confidence=0.85,
            price=current_price,
            quantity=quantity,
            account=account,
        )
        record = self.accept_trade(signal, account=account)

        config = CustomModelConfig(
            symbol=symbol,
            model_name=model_name,
            account=account,
            allocation_amount=allocation_amount,
            parameters={"target_return": 0.10},
        )
        self.custom_models[symbol] = config

        self.alert_manager.send(
            title=f"Trade Alert: {model_name}",
            message=f"{signal.reason} | Action: {signal.action} | Symbol: {signal.symbol} | Qty: {signal.quantity:.4f} | Price: ${signal.price:.2f} | Status: {record.status}",
        )
        return config

    def accept_custom_model(self, symbol: str, suggestion: ModelSuggestion, allocation_amount: float, account: str = "paper") -> CustomModelConfig:
        if allocation_amount <= 0:
            raise ValueError("allocation_amount must be greater than 0")
        if account not in {"paper", "live"}:
            raise ValueError("account must be 'paper' or 'live'")

        config = CustomModelConfig(
            symbol=symbol,
            model_name=suggestion.model_name,
            account=account,
            allocation_amount=allocation_amount,
            parameters=suggestion.parameters,
        )
        self.custom_models[symbol] = config
        return config

    def _evaluate_custom_model_signal(self, data: pd.DataFrame, config: CustomModelConfig) -> Optional[Signal]:
        prepared = self._prepare_history(data)
        if prepared.empty:
            return None

        current_price = float(prepared["Close"].iloc[-1])
        if config.model_name == "MovingAverageCross" and self._rule_moving_average_cross(prepared, len(prepared) - 1):
            return Signal(
                model_name=config.model_name,
                symbol=config.symbol,
                action="buy",
                reason="The 5-day moving average is crossing above the 20-day moving average and the price is above the long-term trend.",
                confidence=0.75,
                price=current_price,
                quantity=config.allocation_amount / current_price,
                account=config.account,
            )

        if config.model_name == "MomentumBreakout" and self._rule_momentum_breakout(prepared, len(prepared) - 1):
            return Signal(
                model_name=config.model_name,
                symbol=config.symbol,
                action="buy",
                reason="Price is breaking above the 20-day high and the short-term momentum is strong enough to justify a buy.",
                confidence=0.8,
                price=current_price,
                quantity=config.allocation_amount / current_price,
                account=config.account,
            )

        if config.model_name == "PullbackAfterMomentum" and self._rule_pullback_after_momentum(
            prepared,
            len(prepared) - 1,
            pullback_pct=float(config.parameters.get("pullback_pct", 0.02)),
            momentum_return=float(config.parameters.get("momentum_return", 0.08)),
            trend_return=float(config.parameters.get("trend_return", 0.12)),
        ):
            return Signal(
                model_name=config.model_name,
                symbol=config.symbol,
                action="buy",
                reason="Price is pulling back toward the 20-day SMA after a strong rally, which matches the selected model.",
                confidence=0.82,
                price=current_price,
                quantity=config.allocation_amount / current_price,
                account=config.account,
            )

        return None

    def scan_custom_model(self, symbol: str, account: str = "paper", historical_data: Optional[pd.DataFrame] = None) -> List[Signal]:
        if symbol not in self.custom_models:
            raise ValueError(f"No custom model has been accepted for {symbol}.")

        config = self.custom_models[symbol]
        if account != config.account:
            raise ValueError(f"Custom model for {symbol} is configured for {config.account}, not {account}.")

        data = historical_data if historical_data is not None else self.fetch_market_data(symbol, period="1y")
        prepared = self._prepare_history(data)
        if prepared.empty:
            return []

        account_state = self.paper_account if account == "paper" else self.live_account
        current_price = float(prepared["Close"].iloc[-1])

        for existing_symbol, position in account_state.positions.items():
            if existing_symbol != symbol:
                continue
            avg_price = position["avg_price"]
            if current_price >= avg_price * 1.10:
                sell_signal = Signal(
                    model_name=config.model_name,
                    symbol=symbol,
                    action="sell",
                    reason=f"{symbol} has moved up {((current_price / avg_price) - 1) * 100:.2f}% from the average entry price of ${avg_price:.2f}.",
                    confidence=0.9,
                    price=current_price,
                    quantity=position["quantity"],
                    account=account,
                )
                self.alert_manager.send(
                    title=f"Trade Alert: {config.model_name}",
                    message=f"{sell_signal.reason} | Action: {sell_signal.action} | Symbol: {sell_signal.symbol} | Qty: {sell_signal.quantity:.4f} | Price: ${sell_signal.price:.2f}",
                )
                return [sell_signal]

        buy_signal = self._evaluate_custom_model_signal(prepared, config)
        if buy_signal is None:
            return []

        self.alert_manager.send(
            title=f"Trade Alert: {config.model_name}",
            message=f"{buy_signal.reason} | Action: {buy_signal.action} | Symbol: {buy_signal.symbol} | Qty: {buy_signal.quantity:.4f} | Price: ${buy_signal.price:.2f}",
        )
        return [buy_signal]

    def run_sp500_scan(self, csv_path: str | Path = "sp500_data.csv", buy_amount: float = 500.0, drop_threshold: float = 2.5) -> List[Signal]:
        data = self.load_csv(csv_path)
        model = SP500DropModel(buy_amount=buy_amount, drop_threshold=drop_threshold)
        signals = model.evaluate(data, "^GSPC")

        for signal in signals:
            self.alert_manager.send(
                title=f"Trade Alert: {signal.model_name}",
                message=f"{signal.reason} | Action: {signal.action} | Symbol: {signal.symbol} | Qty: {signal.quantity:.4f} | Price: ${signal.price:.2f}",
            )
        return signals

    def run_profit_target_scan(self, account: str = "paper", target_return: float = 0.075, market_data_by_symbol: Optional[Dict[str, pd.DataFrame]] = None) -> List[Signal]:
        if account not in {"paper", "live"}:
            raise ValueError("account must be 'paper' or 'live'")

        account_state = self.paper_account if account == "paper" else self.live_account
        market_data_by_symbol = market_data_by_symbol or {}
        signals: List[Signal] = []

        for symbol, position in account_state.positions.items():
            current_price = None
            if symbol in market_data_by_symbol:
                current_data = market_data_by_symbol[symbol]
                current_price = float(current_data["Close"].iloc[-1])
            else:
                current_data = self.fetch_market_data(symbol, period="1y")
                current_price = float(current_data["Close"].iloc[-1])

            avg_price = position["avg_price"]
            if current_price >= avg_price * (1 + target_return):
                quantity = position["quantity"]
                signal = Signal(
                    model_name="ProfitTargetModel",
                    symbol=symbol,
                    action="sell",
                    reason=f"{symbol} has moved up {((current_price / avg_price) - 1) * 100:.2f}% from the average entry price of ${avg_price:.2f}.",
                    confidence=min(1.0, ((current_price / avg_price) - 1) / target_return),
                    price=current_price,
                    quantity=quantity,
                    account=account,
                )
                signals.append(signal)
                self.alert_manager.send(
                    title=f"Trade Alert: {signal.model_name}",
                    message=f"{signal.reason} | Action: {signal.action} | Symbol: {signal.symbol} | Qty: {signal.quantity:.4f} | Price: ${signal.price:.2f}",
                )

        return signals

    def run_trend_scan(self, symbol: str, period: str = "1y") -> List[Signal]:
        data = self.fetch_market_data(symbol, period)
        signals = self.models["trend"].evaluate(data, symbol)
        for signal in signals:
            self.alert_manager.send(
                title=f"Trade Alert: {signal.model_name}",
                message=f"{signal.reason} | Action: {signal.action} | Symbol: {signal.symbol} | Qty: {signal.quantity:.4f} | Price: ${signal.price:.2f}",
            )
        return signals

    def run_diversification_scan(self, symbols: Iterable[str] = ("SPY", "BND", "BTC-USD", "EURUSD=X", "GC=F")) -> List[Signal]:
        all_signals: List[Signal] = []
        for symbol in symbols:
            data = self.fetch_market_data(symbol, period="1y")
            signals = self.models["diversification"].evaluate(data, symbol)
            all_signals.extend(signals)
            for signal in signals:
                self.alert_manager.send(
                    title=f"Trade Alert: {signal.model_name}",
                    message=f"{signal.reason} | Action: {signal.action} | Symbol: {signal.symbol} | Qty: {signal.quantity:.4f} | Price: ${signal.price:.2f}",
                )
        return all_signals

    def scan_model(
        self,
        symbol: str,
        model_name: str,
        account: str = "paper",
        allocation_amount: float = 250.0,
        historical_data: Optional[pd.DataFrame] = None,
        dividend_data: Optional[pd.DataFrame] = None,
    ) -> List[Signal]:
        if account not in {"paper", "live"}:
            raise ValueError("account must be 'paper' or 'live'")

        if symbol in self.custom_models and self.custom_models[symbol].model_name == model_name:
            return self.scan_custom_model(symbol, account=account, historical_data=historical_data)

        data = historical_data if historical_data is not None else self.fetch_market_data(symbol, period="1y")
        prepared = self._prepare_history(data)
        if prepared.empty:
            return []

        if model_name == "DividendCaptureModel":
            resolved_dividend_data = dividend_data if dividend_data is not None else self._fetch_dividend_data(symbol)
            summary = self.models["dividend"].analyze_symbol(symbol, prepared, resolved_dividend_data)
            if summary is None or not summary["eligible"]:
                return []

            signal = Signal(
                model_name="DividendCaptureModel",
                symbol=symbol,
                action="buy",
                reason=f"Dividend capture model selects {symbol} with a {summary['yield_pct'] * 100:.2f}% yield and {summary['days_to_dividend']} days until the next payout.",
                confidence=min(1.0, float(summary["yield_pct"]) / 0.05),
                price=float(summary["current_price"]),
                quantity=allocation_amount / float(summary["current_price"]),
                account=account,
            )
            self.alert_manager.send(
                title=f"Trade Alert: {signal.model_name}",
                message=f"{signal.reason} | Action: {signal.action} | Symbol: {signal.symbol} | Qty: {signal.quantity:.4f} | Price: ${signal.price:.2f}",
            )
            return [signal]

        config = CustomModelConfig(
            symbol=symbol,
            model_name=model_name,
            account=account,
            allocation_amount=allocation_amount,
            parameters={},
        )
        signal = self._evaluate_custom_model_signal(prepared, config)
        if signal is None:
            return []
        self.alert_manager.send(
            title=f"Trade Alert: {signal.model_name}",
            message=f"{signal.reason} | Action: {signal.action} | Symbol: {signal.symbol} | Qty: {signal.quantity:.4f} | Price: ${signal.price:.2f}",
        )
        return [signal]

    def _fetch_dividend_data(self, symbol: str, dividend_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if dividend_data is not None:
            return dividend_data

        try:
            series = yf.Ticker(symbol).dividends
        except Exception:
            return pd.DataFrame(columns=["Date", "Dividends"])

        if series is None or series.empty:
            return pd.DataFrame(columns=["Date", "Dividends"])

        dividend_frame = series.reset_index()
        if dividend_frame.empty:
            return pd.DataFrame(columns=["Date", "Dividends"])
        dividend_frame.columns = ["Date", "Dividends"]
        dividend_frame["Date"] = pd.to_datetime(dividend_frame["Date"], errors="coerce")
        dividend_frame["Dividends"] = pd.to_numeric(dividend_frame["Dividends"], errors="coerce")
        return dividend_frame.dropna(subset=["Date", "Dividends"]).reset_index(drop=True)

    def backtest_dividend_capture(
        self,
        symbols: Iterable[str],
        initial_cash: float = 1000.0,
        allocation_amount: float = 250.0,
        dividend_window: int = 45,
        historical_data_by_symbol: Optional[Dict[str, pd.DataFrame]] = None,
        dividend_data_by_symbol: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> List[Dict[str, object]]:
        del initial_cash
        del allocation_amount
        historical_data_by_symbol = historical_data_by_symbol or {}
        dividend_data_by_symbol = dividend_data_by_symbol or {}
        results: List[Dict[str, object]] = []

        for symbol in symbols:
            history = historical_data_by_symbol.get(symbol)
            if history is None:
                try:
                    history = self.fetch_market_data(symbol, period="1y")
                except Exception:
                    continue

            prepared = self._prepare_history(history)
            if prepared.empty:
                continue

            dividend_data = dividend_data_by_symbol.get(symbol)
            summary = self.models["dividend"].analyze_symbol(symbol, prepared, self._fetch_dividend_data(symbol, dividend_data))
            if summary is None:
                continue

            summary["eligible"] = summary["days_to_dividend"] <= dividend_window and summary["yield_pct"] >= self.models["dividend"].min_yield
            results.append(summary)

        return results

    def run_dividend_capture_scan(
        self,
        symbols: Iterable[str],
        account: str = "paper",
        allocation_amount: float = 250.0,
        dividend_window: int = 45,
        historical_data_by_symbol: Optional[Dict[str, pd.DataFrame]] = None,
        dividend_data_by_symbol: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> List[Signal]:
        if account not in {"paper", "live"}:
            raise ValueError("account must be 'paper' or 'live'")

        account_state = self._get_account_state(account)
        historical_data_by_symbol = historical_data_by_symbol or {}
        dividend_data_by_symbol = dividend_data_by_symbol or {}

        summaries = self.backtest_dividend_capture(
            symbols,
            allocation_amount=allocation_amount,
            dividend_window=dividend_window,
            historical_data_by_symbol=historical_data_by_symbol,
            dividend_data_by_symbol=dividend_data_by_symbol,
        )

        signals: List[Signal] = []
        if not summaries:
            return signals

        candidate_symbols = {summary["symbol"] for summary in summaries}
        current_position_symbol = None
        for symbol in account_state.positions:
            if symbol in candidate_symbols:
                current_position_symbol = symbol
                break

        best_candidate = max([summary for summary in summaries if summary["days_to_dividend"] <= dividend_window], key=lambda item: item["score"], default=None)

        if current_position_symbol is not None:
            current_summary = next((summary for summary in summaries if summary["symbol"] == current_position_symbol), None)
            if (
                current_summary is not None
                and best_candidate is not None
                and not current_summary["hold_is_better"]
                and current_position_symbol != best_candidate["symbol"]
            ):
                sell_signal = Signal(
                    model_name="DividendCaptureModel",
                    symbol=current_position_symbol,
                    action="sell",
                    reason=f"Dividend capture rotation recommends exiting {current_position_symbol} because the yield advantage is stronger than holding the current position.",
                    confidence=0.9,
                    price=float(current_summary["current_price"]),
                    quantity=float(account_state.positions[current_position_symbol]["quantity"]),
                    account=account,
                )
                signals.append(sell_signal)
                self.alert_manager.send(
                    title=f"Trade Alert: {sell_signal.model_name}",
                    message=f"{sell_signal.reason} | Action: {sell_signal.action} | Symbol: {sell_signal.symbol} | Qty: {sell_signal.quantity:.4f} | Price: ${sell_signal.price:.2f}",
                )

        if best_candidate is not None and current_position_symbol != best_candidate["symbol"]:
            history = historical_data_by_symbol.get(best_candidate["symbol"])
            if history is None:
                history = self.fetch_market_data(best_candidate["symbol"], period="1y")
            prepared = self._prepare_history(history)
            current_price = float(prepared["Close"].iloc[-1])
            buy_signal = Signal(
                model_name="DividendCaptureModel",
                symbol=str(best_candidate["symbol"]),
                action="buy",
                reason=f"Dividend capture model selects {best_candidate['symbol']} with a {best_candidate['yield_pct'] * 100:.2f}% yield and {best_candidate['days_to_dividend']} days until the next payout.",
                confidence=min(1.0, float(best_candidate["yield_pct"]) / 0.05),
                price=current_price,
                quantity=allocation_amount / current_price,
                account=account,
            )
            signals.append(buy_signal)
            self.alert_manager.send(
                title=f"Trade Alert: {buy_signal.model_name}",
                message=f"{buy_signal.reason} | Action: {buy_signal.action} | Symbol: {buy_signal.symbol} | Qty: {buy_signal.quantity:.4f} | Price: ${buy_signal.price:.2f}",
            )

        return signals

    def accept_trade(self, signal: Signal, account: str = "paper") -> TradeRecord:
        if account == "paper":
            return self.paper_account.accept_order(signal)
        if account == "live":
            if signal.action in {"buy", "long"}:
                record = TradeRecord(
                    trade_id=f"{account}-{signal.symbol}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
                    account=account,
                    model_name=signal.model_name,
                    symbol=signal.symbol,
                    action=signal.action,
                    quantity=signal.quantity,
                    price=signal.price,
                    status="pending_live",
                    timestamp=_utc_timestamp(),
                    notes="Live order captured; broker execution not connected yet.",
                )
                self.live_ledger.add_record(record)
                return record
            record = TradeRecord(
                trade_id=f"{account}-{signal.symbol}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
                account=account,
                model_name=signal.model_name,
                symbol=signal.symbol,
                action=signal.action,
                quantity=signal.quantity,
                price=signal.price,
                status="pending_live_sell",
                timestamp=_utc_timestamp(),
                notes="Live sell order captured; broker execution not connected yet.",
            )
            self.live_ledger.add_record(record)
            return record
        raise ValueError("account must be 'paper' or 'live'")

    def get_trade_history(self, account: Optional[str] = None) -> List[TradeRecord]:
        if account == "paper":
            return self.paper_ledger.get_records()
        if account == "live":
            return self.live_ledger.get_records()
        return self.paper_ledger.get_records() + self.live_ledger.get_records()

    def _get_account_state(self, account: str):
        if account == "paper":
            return self.paper_account
        if account == "live":
            return self.live_account
        raise ValueError("account must be 'paper' or 'live'")

    def _parse_timestamp(self, timestamp: str) -> datetime:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    def _position_book_value(self, positions: Dict[str, Dict[str, float]]) -> float:
        return sum(position["quantity"] * position["avg_price"] for position in positions.values())

    def _position_market_value(self, positions: Dict[str, Dict[str, float]]) -> float:
        total_value = 0.0
        for symbol, position in positions.items():
            try:
                market_data = self.fetch_market_data(symbol, period="1mo")
                current_price = float(market_data["Close"].iloc[-1])
            except Exception:
                current_price = float(position["avg_price"])
            total_value += current_price * position["quantity"]
        return total_value

    def add_bank_transaction(self, amount: float, account: str = "paper", notes: str = "") -> TradeRecord:
        if account == "paper":
            return self.paper_account.add_cash(amount, notes=notes)
        if account == "live":
            return self.live_account.add_cash(amount, notes=notes)
        raise ValueError("account must be 'paper' or 'live'")

    def get_account_performance(self, account: str = "paper", benchmark_apy: float = 0.04) -> Dict[str, float]:
        account_state = self._get_account_state(account)
        records = self.get_trade_history(account)
        sorted_records = sorted(records, key=lambda record: self._parse_timestamp(record.timestamp))

        total_deposits = sum(record.quantity for record in sorted_records if record.action == "deposit")
        total_withdrawals = sum(record.quantity for record in sorted_records if record.action == "withdraw")
        current_cash = account_state.cash
        current_market_value = self._position_market_value(account_state.positions)
        current_equity = current_cash + current_market_value

        total_inflows = account_state.starting_cash + total_deposits - total_withdrawals
        trading_pnl = current_equity - total_inflows

        now = datetime.now(timezone.utc)
        benchmark_projection = 0.0
        for record in sorted_records:
            if record.action != "deposit":
                continue
            record_dt = self._parse_timestamp(record.timestamp)
            years_elapsed = max((now - record_dt).total_seconds() / (365.25 * 24 * 60 * 60), 0.0)
            benchmark_projection += record.quantity * ((1 + benchmark_apy) ** years_elapsed)

        return {
            "starting_cash": float(account_state.starting_cash),
            "current_cash": float(current_cash),
            "current_equity": float(current_equity),
            "total_deposits": float(total_deposits),
            "total_withdrawals": float(total_withdrawals),
            "trading_pnl": float(trading_pnl),
            "return_pct": float(trading_pnl / total_inflows) if total_inflows else 0.0,
            "benchmark_apy": float(benchmark_apy),
            "benchmark_projection": float(benchmark_projection),
        }

    def get_account_timeline(self, account: str = "paper", benchmark_apy: float = 0.04) -> List[Dict[str, float]]:
        account_state = self._get_account_state(account)
        records = sorted(self.get_trade_history(account), key=lambda record: self._parse_timestamp(record.timestamp))
        if not records:
            return []

        cash = float(account_state.starting_cash)
        positions = {}
        timeline = []
        now = datetime.now(timezone.utc)

        for record in records:
            if record.action == "deposit":
                cash += record.quantity
            elif record.action == "withdraw":
                cash -= record.quantity
            elif record.action in {"buy", "long"}:
                current = positions.get(record.symbol, {"quantity": 0.0, "avg_price": 0.0})
                current["quantity"] += record.quantity
                current["avg_price"] = ((current["avg_price"] * (current["quantity"] - record.quantity)) + record.price * record.quantity) / current["quantity"]
                positions[record.symbol] = current
                cash -= record.quantity * record.price
            elif record.action in {"sell", "short"}:
                current = positions[record.symbol]
                current["quantity"] -= record.quantity
                cash += record.quantity * record.price
                if current["quantity"] == 0:
                    del positions[record.symbol]

            book_value = cash + self._position_book_value(positions)
            benchmark_projection = 0.0
            for deposit_record in records[: records.index(record) + 1]:
                if deposit_record.action != "deposit":
                    continue
                deposit_dt = self._parse_timestamp(deposit_record.timestamp)
                years_elapsed = max((now - deposit_dt).total_seconds() / (365.25 * 24 * 60 * 60), 0.0)
                benchmark_projection += deposit_record.quantity * ((1 + benchmark_apy) ** years_elapsed)

            timeline.append(
                {
                    "timestamp": self._parse_timestamp(record.timestamp).timestamp(),
                    "equity": float(book_value),
                    "benchmark": float(benchmark_projection),
                }
            )

        return timeline

    def get_account_snapshot(self, account: str) -> Dict[str, float | Dict[str, Dict[str, float]]]:
        return self.paper_account.get_snapshot() if account == "paper" else self.live_account.get_snapshot()


if __name__ == "__main__":
    from trading_gui import launch_trading_gui

    launch_trading_gui()
