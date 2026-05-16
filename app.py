import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import urllib.parse
import urllib.request
from pathlib import Path
from math import log, ceil

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except Exception:
    st_autorefresh = None
    AUTOREFRESH_AVAILABLE = False

st.set_page_config(page_title="20-Pip Challenge Confluence Scanner", page_icon="🎯", layout="wide")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
TELEGRAM_SENT_LOG_FILE = DATA_DIR / "telegram_sent_signals.csv"
ACTIVE_SIGNAL_STATE_FILE = DATA_DIR / "active_signal_state.csv"
CHALLENGE_STATE_FILE = DATA_DIR / "challenge_state.csv"

PAIRS = {
    "EURUSD": {"ticker": "EURUSD=X", "pip": 0.0001, "decimals": 5, "default": 1.10000},
    "GBPUSD": {"ticker": "GBPUSD=X", "pip": 0.0001, "decimals": 5, "default": 1.27000},
    "USDJPY": {"ticker": "USDJPY=X", "pip": 0.01, "decimals": 3, "default": 150.000},
    "USDCAD": {"ticker": "USDCAD=X", "pip": 0.0001, "decimals": 5, "default": 1.35000},
    # Optional: confirm your broker's XAUUSD pip convention before trading gold.
    "XAUUSD": {"ticker": "GC=F", "pip": 0.01, "decimals": 2, "default": 2350.00},
}
DEFAULT_SCAN_MARKETS = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD"]
STRATEGY_NAME = "Robust 20-Pip 1H/15M Confluence"

BUY_BG = "#dcfce7"
SELL_BG = "#fee2e2"
NEUTRAL_BG = "#ffedd5"
LOCK_BG = "#e0e7ff"

ACTIVE_COLUMNS = [
    "Market", "Signal_ID", "Strategy", "Direction", "Entry", "SL", "TP",
    "Opened_At", "Opened_Candle_Date", "Status", "Closed_At", "Close_Reason", "Close_Price"
]
LOG_COLUMNS = ["Signal_Key", "Signal_ID", "Timestamp", "Strategy", "Market", "Direction", "Message"]


def get_secret_value(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def safe_timestamp():
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_price(market, value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{PAIRS[market]['decimals']}f}"


def pips_to_price(market, pips):
    return float(pips) * PAIRS[market]["pip"]


def current_utc_time():
    return pd.Timestamp.now(tz="UTC")


def is_market_open(market, now_utc=None):
    if now_utc is None:
        now_utc = current_utc_time()
    weekday = now_utc.weekday()  # Monday=0, Sunday=6
    hour = now_utc.hour + now_utc.minute / 60
    if weekday == 5:
        return False, "CLOSED_WEEKEND"
    if weekday == 6 and hour < 22:
        return False, "CLOSED_WEEKEND"
    if weekday == 4 and hour >= 22:
        return False, "CLOSED_WEEKEND"
    return True, "OPEN"


def candle_age_minutes(candle_time):
    try:
        candle_ts = pd.Timestamp(candle_time)
        if candle_ts.tzinfo is None:
            candle_ts = candle_ts.tz_localize("UTC")
        else:
            candle_ts = candle_ts.tz_convert("UTC")
        return round((current_utc_time() - candle_ts).total_seconds() / 60, 1)
    except Exception:
        return np.nan


def is_fresh_signal_candle(candle_time, max_age_minutes):
    age = candle_age_minutes(candle_time)
    if pd.isna(age):
        return False, age, "UNKNOWN_CANDLE_TIME"
    if age > max_age_minutes:
        return False, age, f"STALE_{age:.1f}_MIN"
    return True, age, "FRESH"


def add_cameroon_time(df, start_hour, end_hour):
    df = df.copy()
    local = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert("Africa/Douala")
    df["Cameroon_Time"] = local.dt.strftime("%Y-%m-%d %H:%M")
    hour = local.dt.hour + local.dt.minute / 60
    if start_hour < end_hour:
        df["Trading_Window"] = (hour >= start_hour) & (hour < end_hour)
    else:
        df["Trading_Window"] = (hour >= start_hour) | (hour < end_hour)
    return df


def color_rows(df):
    def apply(row):
        sig = str(row.get("Signal", "")).upper()
        active = str(row.get("Active_Signal_Status", "")).upper()
        if active == "OPEN":
            return [f"background-color: {LOCK_BG}; color: #3730a3"] * len(row)
        if sig == "BUY":
            return [f"background-color: {BUY_BG}; color: #065f46"] * len(row)
        if sig == "SELL":
            return [f"background-color: {SELL_BG}; color: #991b1b"] * len(row)
        return [f"background-color: {NEUTRAL_BG}; color: #9a3412"] * len(row)
    return df.style.apply(apply, axis=1)


# ============================ DATA ============================

def clean_data(raw):
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index()
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    if "Volume" not in df.columns:
        df["Volume"] = 0
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(None)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data(ttl=300)
def load_market_data(market, scan_speed="Fast"):
    ticker = PAIRS[market]["ticker"]
    if scan_speed == "Fast":
        h1_period, m15_period = "90d", "10d"
    elif scan_speed == "Balanced":
        h1_period, m15_period = "180d", "30d"
    else:
        h1_period, m15_period = "730d", "60d"
    h1_raw = yf.download(ticker, period=h1_period, interval="1h", progress=False, auto_adjust=False, threads=False, timeout=20)
    m15_raw = yf.download(ticker, period=m15_period, interval="15m", progress=False, auto_adjust=False, threads=False, timeout=20)
    return clean_data(h1_raw), clean_data(m15_raw)


# ============================ INDICATORS ============================

def add_ema(df):
    df = df.copy()
    for span in [8, 20, 50, 200]:
        df[f"EMA_{span}"] = df["Close"].ewm(span=span, adjust=False).mean()
    return df


def add_rsi(df, period=14):
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df):
    df = df.copy()
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_atr(df, period=14):
    df = df.copy()
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df[f"ATR_{period}"] = tr.rolling(period).mean()
    return df


def add_adx(df, period=14):
    df = df.copy()
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    tr_sum = tr.rolling(period).sum()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).sum() / tr_sum
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).sum() / tr_sum
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df[f"ADX_{period}"] = dx.rolling(period).mean()
    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di
    return df


def add_indicators(df):
    return add_adx(add_atr(add_macd(add_rsi(add_ema(df)))))


# ============================ ACTIVE SIGNAL LOCK ============================

def load_active_signal_state():
    if ACTIVE_SIGNAL_STATE_FILE.exists():
        try:
            df = pd.read_csv(ACTIVE_SIGNAL_STATE_FILE)
            for col in ACTIVE_COLUMNS:
                if col not in df.columns:
                    df[col] = np.nan
            return df[ACTIVE_COLUMNS]
        except Exception:
            return pd.DataFrame(columns=ACTIVE_COLUMNS)
    return pd.DataFrame(columns=ACTIVE_COLUMNS)


def save_active_signal_state(df):
    df = df.copy()
    for col in ACTIVE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df[ACTIVE_COLUMNS].to_csv(ACTIVE_SIGNAL_STATE_FILE, index=False)


def clear_active_signal_state():
    if ACTIVE_SIGNAL_STATE_FILE.exists():
        ACTIVE_SIGNAL_STATE_FILE.unlink()


def get_open_active_signal(market=None):
    """
    Global active trade lock.
    Only one active trade is allowed across all pairs.
    """
    df = load_active_signal_state()
    if df.empty:
        return None

    rows = df[df["Status"].astype(str) == "OPEN"]

    if rows.empty:
        return None

    return rows.iloc[-1].to_dict()


def active_signal_blocks_new_signal(market=None):
    active = get_open_active_signal()
    if active is None:
        return False, ""

    return True, (
        f"One active trade is already OPEN: {active.get('Market')} {active.get('Direction')} "
        f"at Level {active.get('Challenge_Level', '')}. Waiting for TP {active.get('TP')} "
        f"or SL {active.get('SL')} before allowing any new signal."
    )


def record_active_signal(signal_id, strategy, market, direction, entry, sl, tp, opened_candle_date, challenge_level=1):
    df = load_active_signal_state()
    rows = df[(df["Market"].astype(str) == str(market)) & (df["Status"].astype(str) == "OPEN")]
    if not rows.empty:
        return
    new = pd.DataFrame([{
        "Market": market,
        "Signal_ID": signal_id,
        "Strategy": strategy,
        "Direction": direction,
        "Entry": float(entry),
        "SL": float(sl),
        "TP": float(tp),
        "Opened_At": safe_timestamp(),
        "Opened_Candle_Date": str(opened_candle_date),
        "Status": "OPEN",
        "Closed_At": "",
        "Close_Reason": "",
        "Close_Price": np.nan,
        "Challenge_Level": int(challenge_level),
    }])
    save_active_signal_state(pd.concat([df, new], ignore_index=True))


def evaluate_signal_outcome_from_candles(direction, sl, tp, candles):
    if candles is None or candles.empty:
        return None, None, None
    for _, candle in candles.iterrows():
        high = pd.to_numeric(candle.get("High", np.nan), errors="coerce")
        low = pd.to_numeric(candle.get("Low", np.nan), errors="coerce")
        if pd.isna(high) or pd.isna(low):
            continue
        if direction == "BUY":
            if float(low) <= float(sl):
                return "SL", float(sl), candle.get("Date", "")
            if float(high) >= float(tp):
                return "TP", float(tp), candle.get("Date", "")
        if direction == "SELL":
            if float(high) >= float(sl):
                return "SL", float(sl), candle.get("Date", "")
            if float(low) <= float(tp):
                return "TP", float(tp), candle.get("Date", "")
    return None, None, None


def update_open_signal_outcomes(scanner_cache):
    df = load_active_signal_state()
    statuses = []
    if df.empty:
        return statuses
    changed = False
    for idx, row in df[df["Status"].astype(str) == "OPEN"].iterrows():
        market = str(row.get("Market", ""))
        m15 = scanner_cache.get(market, {}).get("m15", pd.DataFrame()) if isinstance(scanner_cache, dict) else pd.DataFrame()
        if m15 is None or m15.empty:
            continue
        candles = m15.copy()
        candles["Date"] = pd.to_datetime(candles["Date"], errors="coerce")
        opened = pd.to_datetime(row.get("Opened_Candle_Date", ""), errors="coerce")
        candles_to_check = candles[candles["Date"] >= opened].copy() if pd.notna(opened) else candles.tail(20).copy()
        if candles_to_check.empty:
            candles_to_check = candles.tail(20).copy()
        outcome, close_price, _ = evaluate_signal_outcome_from_candles(row.get("Direction"), row.get("SL"), row.get("TP"), candles_to_check)
        if outcome in ["TP", "SL"]:
            df.loc[idx, "Status"] = outcome
            df.loc[idx, "Closed_At"] = safe_timestamp()
            df.loc[idx, "Close_Reason"] = "Take Profit hit" if outcome == "TP" else "Stop Loss hit"
            df.loc[idx, "Close_Price"] = close_price
            statuses.append(f"{market}: previous {row.get('Direction')} signal closed by {outcome} at {close_price}. New signal allowed.")
            changed = True
    if changed:
        save_active_signal_state(df)
    return statuses


def apply_active_signal_lock(scanner_df):
    if scanner_df is None or scanner_df.empty:
        return scanner_df
    df = scanner_df.copy()
    df["Active_Signal_Status"] = "NONE"
    for idx, row in df.iterrows():
        blocked, msg = active_signal_blocks_new_signal(row.get("Market", ""))
        if blocked:
            df.loc[idx, "Active_Signal_Status"] = "OPEN"
            if row.get("Signal") in ["BUY", "SELL"]:
                df.loc[idx, "Signal"] = "NEUTRAL"
                df.loc[idx, "Direction"] = "NEUTRAL"
                df.loc[idx, "Reason"] = msg
    return df


# ============================ TELEGRAM ============================

def load_telegram_sent_log():
    if TELEGRAM_SENT_LOG_FILE.exists():
        try:
            df = pd.read_csv(TELEGRAM_SENT_LOG_FILE)
            for col in LOG_COLUMNS:
                if col not in df.columns:
                    df[col] = np.nan
            return df[LOG_COLUMNS]
        except Exception:
            return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.DataFrame(columns=LOG_COLUMNS)


def save_telegram_sent_signal(signal_key, signal_id, strategy, market, direction, message):
    log = load_telegram_sent_log()
    new = pd.DataFrame([{
        "Signal_Key": signal_key,
        "Signal_ID": signal_id,
        "Timestamp": safe_timestamp(),
        "Strategy": strategy,
        "Market": market,
        "Direction": direction,
        "Message": message,
    }])
    pd.concat([log, new], ignore_index=True).to_csv(TELEGRAM_SENT_LOG_FILE, index=False)


def clear_telegram_sent_log():
    if TELEGRAM_SENT_LOG_FILE.exists():
        TELEGRAM_SENT_LOG_FILE.unlink()


def already_sent_exact_signal(signal_id):
    log = load_telegram_sent_log()
    if log.empty or "Signal_ID" not in log.columns:
        return False
    return str(signal_id) in set(log["Signal_ID"].astype(str))


def make_signal_key(strategy, market, direction):
    return f"{strategy}|{market}|{direction}"


def make_signal_id(strategy, market, direction, entry, sl, tp, source_time=""):
    return f"{strategy}|{market}|{direction}|{round(float(entry), 8)}|{round(float(sl), 8)}|{round(float(tp), 8)}|{source_time}"


def send_telegram_message(bot_token, chat_id, message):
    bot_token = str(bot_token).strip()
    chat_id = str(chat_id).strip()
    if not bot_token:
        return False, "Telegram BOT_TOKEN is missing."
    if not chat_id:
        return False, "Telegram CHAT_ID is missing."
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }).encode("utf-8")
        request = urllib.request.Request(url, data=payload, method="POST")
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8", errors="ignore")
        return (True, "Telegram message sent successfully.") if '"ok":true' in body else (False, f"Telegram API returned: {body[:300]}")
    except Exception as e:
        return False, f"Telegram send failed: {e}"


def format_telegram_signal_message(strategy, market, direction, entry, sl, tp, target_pips, stop_pips, confluence_score, cameroon_time="", reason=""):
    emoji = "🟢" if direction == "BUY" else "🔴"
    lines = [
        f"{emoji} <b>20-PIP CHALLENGE SIGNAL</b>",
        "",
        f"<b>Strategy:</b> {strategy}",
        f"<b>Market:</b> {market}",
        f"<b>Direction:</b> {direction}",
        f"<b>Entry:</b> {entry}",
        f"<b>Stop Loss:</b> {sl}",
        f"<b>Take Profit:</b> {tp}",
        f"<b>Target:</b> {target_pips} pips",
        f"<b>Stop:</b> {stop_pips} pips",
        f"<b>Confluence:</b> {confluence_score}",
    ]
    if cameroon_time:
        lines.append(f"<b>Cameroon Time:</b> {cameroon_time}")
    if reason:
        lines.extend(["", f"<b>Reason:</b> {reason}"])
    lines.extend(["", "⚠️ High-risk challenge. Confirm broker price, spread, liquidity, and news before trading."])
    return "\n".join(lines)


# ============================ STRATEGY ============================

def rolling_pullback_buy(df, lookback=6):
    return (df["Low"].rolling(lookback).min() <= df["EMA_20"].rolling(lookback).max()) & (df["Close"] > df["EMA_20"])


def rolling_pullback_sell(df, lookback=6):
    return (df["High"].rolling(lookback).max() >= df["EMA_20"].rolling(lookback).min()) & (df["Close"] < df["EMA_20"])


def build_20pip_strategy(h1, m15, market, target_pips=20, stop_pips=10, min_adx=20, min_atr_pips=3, breakout_lookback=8, pullback_lookback=6, session_start=7, session_end=18, enforce_session=True, strict_all=True, max_signal_age_minutes=90):
    h1 = add_indicators(h1)
    m15 = add_indicators(m15)
    h1["H1_Buy_Trend"] = (h1["EMA_50"] > h1["EMA_200"]) & (h1["Close"] > h1["EMA_50"])
    h1["H1_Sell_Trend"] = (h1["EMA_50"] < h1["EMA_200"]) & (h1["Close"] < h1["EMA_50"])
    h1["H1_Buy_Momentum"] = (h1["RSI_14"] > 50) & (h1["MACD_Hist"] > 0) & (h1["PLUS_DI"] > h1["MINUS_DI"])
    h1["H1_Sell_Momentum"] = (h1["RSI_14"] < 50) & (h1["MACD_Hist"] < 0) & (h1["MINUS_DI"] > h1["PLUS_DI"])
    h1["H1_ADX_OK"] = h1["ADX_14"] >= min_adx
    state = h1[["Date", "H1_Buy_Trend", "H1_Sell_Trend", "H1_Buy_Momentum", "H1_Sell_Momentum", "H1_ADX_OK", "ADX_14"]].dropna().sort_values("Date")
    m15 = m15.sort_values("Date").reset_index(drop=True)
    if not state.empty:
        m15 = pd.merge_asof(m15, state, on="Date", direction="backward")
    else:
        for c in ["H1_Buy_Trend", "H1_Sell_Trend", "H1_Buy_Momentum", "H1_Sell_Momentum", "H1_ADX_OK"]:
            m15[c] = False
        m15["ADX_14"] = np.nan

    m15["M15_Buy_Trend"] = (m15["EMA_8"] > m15["EMA_20"]) & (m15["EMA_20"] > m15["EMA_50"]) & (m15["Close"] > m15["EMA_20"])
    m15["M15_Sell_Trend"] = (m15["EMA_8"] < m15["EMA_20"]) & (m15["EMA_20"] < m15["EMA_50"]) & (m15["Close"] < m15["EMA_20"])
    m15["M15_Buy_Pullback"] = rolling_pullback_buy(m15, pullback_lookback)
    m15["M15_Sell_Pullback"] = rolling_pullback_sell(m15, pullback_lookback)
    prev_high = m15["High"].rolling(breakout_lookback).max().shift(1)
    prev_low = m15["Low"].rolling(breakout_lookback).min().shift(1)
    m15["M15_Buy_Breakout"] = m15["Close"] > prev_high
    m15["M15_Sell_Breakout"] = m15["Close"] < prev_low
    m15["M15_Buy_Momentum"] = (m15["RSI_14"] > 52) & (m15["MACD_Hist"] > 0)
    m15["M15_Sell_Momentum"] = (m15["RSI_14"] < 48) & (m15["MACD_Hist"] < 0)
    m15["ATR_Pips"] = m15["ATR_14"] / PAIRS[market]["pip"]
    m15["ATR_OK"] = m15["ATR_Pips"] >= min_atr_pips
    m15 = add_cameroon_time(m15, session_start, session_end)
    market_open, market_status = is_market_open(market)
    m15["Market_Open"] = market_open
    m15["Market_Status"] = market_status
    m15["Fresh_OK"] = False
    m15["Candle_Age_Min"] = np.nan
    m15["Freshness"] = ""
    if not m15.empty:
        i = m15.index[-1]
        ok, age, freshness = is_fresh_signal_candle(m15.loc[i, "Date"], max_signal_age_minutes)
        m15.loc[i, "Fresh_OK"] = ok
        m15.loc[i, "Candle_Age_Min"] = age
        m15.loc[i, "Freshness"] = freshness

    buy_checks = ["H1_Buy_Trend", "H1_Buy_Momentum", "H1_ADX_OK", "M15_Buy_Trend", "M15_Buy_Pullback", "M15_Buy_Breakout", "M15_Buy_Momentum", "ATR_OK", "Market_Open", "Fresh_OK"]
    sell_checks = ["H1_Sell_Trend", "H1_Sell_Momentum", "H1_ADX_OK", "M15_Sell_Trend", "M15_Sell_Pullback", "M15_Sell_Breakout", "M15_Sell_Momentum", "ATR_OK", "Market_Open", "Fresh_OK"]
    if enforce_session:
        buy_checks.append("Trading_Window")
        sell_checks.append("Trading_Window")
    if strict_all:
        buy = pd.Series(True, index=m15.index)
        sell = pd.Series(True, index=m15.index)
        for c in buy_checks:
            buy &= m15[c].astype(bool)
        for c in sell_checks:
            sell &= m15[c].astype(bool)
    else:
        buy = sum(m15[c].astype(bool).astype(int) for c in buy_checks) >= max(8, len(buy_checks) - 1)
        sell = sum(m15[c].astype(bool).astype(int) for c in sell_checks) >= max(8, len(sell_checks) - 1)

    m15["Buy_Confluence"] = sum(m15[c].astype(bool).astype(int) for c in buy_checks)
    m15["Sell_Confluence"] = sum(m15[c].astype(bool).astype(int) for c in sell_checks)
    m15["Required_Confluence"] = len(buy_checks)
    m15["Signal"] = "NEUTRAL"
    m15["Direction"] = "NEUTRAL"
    m15["Reason"] = "No complete confluence"
    m15["Entry"] = np.nan
    m15["SL"] = np.nan
    m15["TP"] = np.nan
    target_distance = pips_to_price(market, target_pips)
    stop_distance = pips_to_price(market, stop_pips)
    m15.loc[buy, ["Signal", "Direction", "Reason", "Entry", "SL", "TP"]] = ["BUY", "BUY", "All BUY confluences valid", m15["Close"], m15["Close"] - stop_distance, m15["Close"] + target_distance]
    m15.loc[sell, ["Signal", "Direction", "Reason", "Entry", "SL", "TP"]] = ["SELL", "SELL", "All SELL confluences valid", m15["Close"], m15["Close"] + stop_distance, m15["Close"] - target_distance]
    return h1, m15


def scan_markets(markets, scan_speed, target_pips, stop_pips, min_adx, min_atr_pips, breakout_lookback, pullback_lookback, session_start, session_end, enforce_session, strict_all, max_signal_age_minutes):
    rows, cache = [], {}
    for market in markets:
        try:
            h1_raw, m15_raw = load_market_data(market, scan_speed)
            if h1_raw.empty or m15_raw.empty:
                rows.append({"Market": market, "Signal": "NEUTRAL", "Direction": "NEUTRAL", "Reason": "Not enough data", "Entry": np.nan, "SL": np.nan, "TP": np.nan})
                cache[market] = {"h1": h1_raw, "m15": m15_raw}
                continue
            h1, m15 = build_20pip_strategy(h1_raw, m15_raw, market, target_pips, stop_pips, min_adx, min_atr_pips, breakout_lookback, pullback_lookback, session_start, session_end, enforce_session, strict_all, max_signal_age_minutes)
            cache[market] = {"h1": h1, "m15": m15}
            latest = m15.dropna(subset=["Close"]).iloc[-1].to_dict()
            rows.append({
                "Market": market,
                "Signal": latest.get("Signal", "NEUTRAL"),
                "Direction": latest.get("Direction", "NEUTRAL"),
                "Signal_Date": str(latest.get("Date", "")),
                "Entry": latest.get("Entry", np.nan),
                "SL": latest.get("SL", np.nan),
                "TP": latest.get("TP", np.nan),
                "Last_Close": latest.get("Close", np.nan),
                "Cameroon_Time": latest.get("Cameroon_Time", "N/A"),
                "Trading_Window": bool(latest.get("Trading_Window", False)),
                "Market_Status": latest.get("Market_Status", "UNKNOWN"),
                "Freshness": latest.get("Freshness", ""),
                "Candle_Age_Min": latest.get("Candle_Age_Min", np.nan),
                "Buy_Confluence": latest.get("Buy_Confluence", 0),
                "Sell_Confluence": latest.get("Sell_Confluence", 0),
                "Required_Confluence": latest.get("Required_Confluence", 0),
                "H1_ADX": latest.get("ADX_14", np.nan),
                "ATR_Pips": latest.get("ATR_Pips", np.nan),
                "Reason": latest.get("Reason", ""),
            })
        except Exception as e:
            rows.append({"Market": market, "Signal": "NEUTRAL", "Direction": "NEUTRAL", "Reason": str(e), "Entry": np.nan, "SL": np.nan, "TP": np.nan})
            cache[market] = {"h1": pd.DataFrame(), "m15": pd.DataFrame()}
    scanner_df = pd.DataFrame(rows)
    outcomes = update_open_signal_outcomes(cache)
    scanner_df = apply_active_signal_lock(scanner_df)
    return scanner_df, cache, outcomes


def send_scanner_telegram_alerts(scanner_df, strategy_name, bot_token, chat_id, enable_telegram, auto_send, target_pips, stop_pips):
    statuses = []
    if not enable_telegram or not auto_send or scanner_df is None or scanner_df.empty:
        return statuses
    active = scanner_df[scanner_df["Signal"].isin(["BUY", "SELL"])].copy()
    for _, row in active.iterrows():
        market, direction = row["Market"], row["Direction"]
        entry, sl, tp = row["Entry"], row["SL"], row["TP"]
        cameroon_time, signal_date = str(row.get("Cameroon_Time", "")), str(row.get("Signal_Date", ""))
        blocked, msg = active_signal_blocks_new_signal(market)
        if blocked:
            statuses.append(f"{market}: {msg}")
            continue
        if pd.isna(entry) or pd.isna(sl) or pd.isna(tp):
            statuses.append(f"{market}: Entry/SL/TP missing; skipped.")
            continue
        signal_key = make_signal_key(strategy_name, market, direction)
        signal_id = make_signal_id(strategy_name, market, direction, entry, sl, tp, cameroon_time)
        if already_sent_exact_signal(signal_id):
            statuses.append(f"{market}: exact same candle signal already sent; skipped.")
            continue
        score = row.get("Buy_Confluence", 0) if direction == "BUY" else row.get("Sell_Confluence", 0)
        required = row.get("Required_Confluence", 0)
        msg_text = format_telegram_signal_message(strategy_name, market, direction, fmt_price(market, entry), fmt_price(market, sl), fmt_price(market, tp), target_pips, stop_pips, f"{int(score)}/{int(required)}", cameroon_time, str(row.get("Reason", "")))
        ok, response = send_telegram_message(bot_token, chat_id, msg_text)
        if ok:
            save_telegram_sent_signal(signal_key, signal_id, strategy_name, market, direction, response)
            record_active_signal(signal_id, strategy_name, market, direction, entry, sl, tp, signal_date)
            statuses.append(f"{market}: signal locked as OPEN until TP or SL is hit.")
        statuses.append(f"{market}: {response}")
    return statuses


# ============================ CHARTS ============================

def candle_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
    for col in ["EMA_20", "EMA_50", "EMA_200"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[col], mode="lines", name=col))
    if "Signal" in df.columns:
        buys, sells = df[df["Signal"] == "BUY"], df[df["Signal"] == "SELL"]
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys["Date"], y=buys["Entry"], mode="markers", name="BUY", marker=dict(size=12, symbol="triangle-up", color="green")))
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells["Date"], y=sells["Entry"], mode="markers", name="SELL", marker=dict(size=12, symbol="triangle-down", color="red")))
    fig.update_layout(title=title, height=560, xaxis_rangeslider_visible=False)
    return fig


def challenge_projection(start_balance, target_balance, risk_percent, target_pips, stop_pips):
    if start_balance <= 0 or target_balance <= start_balance or risk_percent <= 0 or stop_pips <= 0:
        return None
    rr = target_pips / stop_pips
    gain_per_win = (risk_percent / 100) * rr
    if gain_per_win <= 0:
        return None
    wins_needed = ceil(log(target_balance / start_balance) / log(1 + gain_per_win))
    return {"RR": rr, "Gain per winning trade": f"{gain_per_win * 100:.2f}%", "Perfect wins needed": wins_needed}


# ============================ APP ============================

st.title("🎯 Robust 20-Pip Challenge Confluence Scanner")
st.caption("15M entry + 1H confirmation. A pair gives a new signal only after the previous signal hits TP or SL.")

with st.sidebar:
    st.header("20-Pip Challenge Controls")
    scan_markets_selected = st.multiselect("Pairs to scan", list(PAIRS.keys()), default=DEFAULT_SCAN_MARKETS)
    chart_market = st.selectbox("Chart market", list(PAIRS.keys()), index=0)
    scan_speed = st.selectbox("Scan speed", ["Fast", "Balanced", "Full"], index=0)

    st.divider()
    st.subheader("Challenge Ladder Pattern")
    start_balance = st.number_input("Level 1 starting balance", min_value=1.0, value=20.0, step=1.0)
    challenge_levels = st.number_input("Number of levels", min_value=5, max_value=100, value=30, step=1)
    risk_percent = st.number_input("Percentage growth/risk pattern", min_value=1.0, max_value=100.0, value=30.0, step=1.0)
    target_pips = st.number_input("Take profit target in pips", min_value=5, value=20, step=1)
    stop_pips = st.number_input("Stop loss in pips", min_value=3, value=10, step=1)
    pip_value_per_lot = st.number_input("Pip value per 1.00 lot", min_value=1.0, value=10.0, step=1.0)

    manual_level = st.number_input("Manual current level", min_value=1, max_value=int(challenge_levels), value=load_challenge_state()["Current_Level"], step=1)
    if st.button("Set current challenge level"):
        set_challenge_level(manual_level)
        st.success(f"Challenge level set to {manual_level}.")

    if st.button("Reset challenge to Level 1"):
        reset_challenge_state()
        st.success("Challenge reset to Level 1 and active trade cleared.")

    st.divider()
    st.subheader("Confluence Rules")
    strict_all = st.checkbox("Require every confluence", value=True)
    min_adx = st.selectbox("Minimum 1H ADX", [15, 18, 20, 22, 25, 30], index=2)
    min_atr_pips = st.selectbox("Minimum 15M ATR in pips", [2, 3, 4, 5, 6, 8, 10], index=1)
    breakout_lookback = st.selectbox("15M breakout lookback", [5, 8, 10, 12, 15], index=1)
    pullback_lookback = st.selectbox("15M pullback lookback", [3, 5, 6, 8, 10], index=2)

    st.divider()
    st.subheader("Session and Freshness")
    enforce_session = st.checkbox("Trade only inside session", value=True)
    session_start = st.selectbox("Session start", list(range(0, 24)), index=7, format_func=lambda x: f"{x:02d}:00 Cameroon")
    session_end = st.selectbox("Session end", list(range(1, 25)), index=17, format_func=lambda x: f"{x if x < 24 else 0:02d}:00 Cameroon")
    max_signal_age_minutes = st.selectbox("Maximum signal candle age", [30, 60, 90, 120], index=1)

    st.divider()
    st.subheader("Telegram")
    enable_telegram = st.checkbox("Enable Telegram alerts", value=True)
    auto_send_telegram = st.checkbox("Auto-send valid signals to Telegram", value=True)
    bot_token = get_secret_value("TELEGRAM_BOT_TOKEN", "")
    chat_id = get_secret_value("TELEGRAM_CHAT_ID", "")
    if bot_token and chat_id:
        st.success("Telegram secrets loaded.")
    else:
        st.warning("Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in Streamlit Secrets.")
    if st.button("Send Telegram Test Message"):
        ok, msg = send_telegram_message(bot_token, chat_id, "✅ Telegram test from Robust 20-Pip Challenge Scanner.")
        st.success(msg) if ok else st.error(msg)
    if st.button("Clear Telegram sent-signal log"):
        clear_telegram_sent_log(); st.success("Telegram sent-signal log cleared.")
    if st.button("Clear active/open signal state"):
        clear_active_signal_state(); st.success("Active/open signal state cleared.")

    st.divider()
    st.subheader("Auto Refresh")
    auto_refresh = st.checkbox("Enable auto-refresh", value=True)
    refresh_minutes = st.selectbox("Refresh every", [3, 5, 10, 15], index=1)
    if auto_refresh:
        if AUTOREFRESH_AVAILABLE and st_autorefresh is not None:
            st_autorefresh(interval=refresh_minutes * 60 * 1000, key="robust_20pip_autorefresh")
            st.caption(f"Auto-refresh active every {refresh_minutes} minute(s).")
        else:
            st.warning("streamlit-autorefresh is not installed.")
    if st.button("Refresh now"):
        st.cache_data.clear(); st.rerun()

if not scan_markets_selected:
    st.warning("Select at least one pair to scan.")
    st.stop()

projection = challenge_projection(start_balance, target_balance, risk_percent, target_pips, stop_pips)
if projection:
    pcols = st.columns(4)
    pcols[0].metric("Challenge", f"${start_balance:,.0f} → ${target_balance:,.0f}")
    pcols[1].metric("TP / SL", f"{target_pips} / {stop_pips} pips")
    pcols[2].metric("Reward-to-risk", f"{projection['RR']:.2f}R")
    pcols[3].metric("Perfect wins needed", projection["Perfect wins needed"])
    st.warning("Projection assumes zero losses and perfect compounding. It is not a guarantee and is extremely high risk.")

scanner_df, scanner_cache, outcome_statuses = scan_markets(
    markets=scan_markets_selected,
    scan_speed=scan_speed,
    target_pips=target_pips,
    stop_pips=stop_pips,
    min_adx=min_adx,
    min_atr_pips=min_atr_pips,
    breakout_lookback=breakout_lookback,
    pullback_lookback=pullback_lookback,
    session_start=session_start,
    session_end=session_end,
    enforce_session=enforce_session,
    strict_all=strict_all,
    max_signal_age_minutes=max_signal_age_minutes,
)
scanner_df["Challenge_Level"] = current_level
scanner_df["Pattern_Lot_Size"] = current_step["Lot_Size"]
scanner_df["Profit_Goal"] = current_step["Profit_Goal"]
scanner_df["Pattern_Risk"] = current_step["Risk"]

telegram_statuses = send_scanner_telegram_alerts(scanner_df, STRATEGY_NAME, bot_token, chat_id, enable_telegram, auto_send_telegram, target_pips, stop_pips)

if outcome_statuses:
    st.subheader("TP/SL Outcome Updates")
    for item in outcome_statuses:
        st.success(item)
if telegram_statuses:
    st.subheader("Telegram Status")
    for item in telegram_statuses:
        st.info(item)

st.subheader("Challenge Ladder Table")
ladder_view = challenge_ladder.copy()
st.dataframe(ladder_view, use_container_width=True)

st.subheader("Scanner Results")
display_df = scanner_df.copy()
for col in ["Entry", "SL", "TP", "Last_Close"]:
    if col in display_df.columns:
        display_df[col] = display_df.apply(lambda r: fmt_price(r["Market"], r[col]) if pd.notna(r[col]) else "N/A", axis=1)
for col in ["H1_ADX", "ATR_Pips", "Candle_Age_Min"]:
    if col in display_df.columns:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(2)
st.dataframe(color_rows(display_df), use_container_width=True)

open_state = load_active_signal_state()
open_state = open_state[open_state["Status"].astype(str) == "OPEN"] if not open_state.empty else open_state
if not open_state.empty:
    st.subheader("Open Signal State")
    st.caption("A pair remains locked until its TP or SL is hit.")
    st.dataframe(open_state, use_container_width=True)

st.divider()
st.subheader(f"{chart_market} 15M Chart and Confluences")
cache_item = scanner_cache.get(chart_market, {})
m15 = cache_item.get("m15", pd.DataFrame())
if m15 is None or m15.empty:
    st.warning(f"No data available for {chart_market}.")
else:
    st.plotly_chart(candle_chart(m15.tail(250), f"{chart_market} 15M - Robust 20-Pip Confluence"), use_container_width=True)
    confluence_cols = [
        "Date", "Cameroon_Time", "Signal", "Direction", "Entry", "SL", "TP",
        "Buy_Confluence", "Sell_Confluence", "Required_Confluence",
        "H1_Buy_Trend", "H1_Sell_Trend", "H1_Buy_Momentum", "H1_Sell_Momentum",
        "H1_ADX_OK", "M15_Buy_Trend", "M15_Sell_Trend", "M15_Buy_Pullback", "M15_Sell_Pullback",
        "M15_Buy_Breakout", "M15_Sell_Breakout", "M15_Buy_Momentum", "M15_Sell_Momentum",
        "ATR_OK", "Trading_Window", "Market_Open", "Fresh_OK", "ATR_Pips", "Reason"
    ]
    confluence_cols = [c for c in confluence_cols if c in m15.columns]
    st.dataframe(color_rows(m15.tail(40)[confluence_cols]), use_container_width=True)

st.warning("Educational tool only. No strategy can guarantee $20 to $54,000. Test on demo first and confirm every signal with your broker prices and news risk.")
