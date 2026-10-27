import os
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf

TOKEN = os.environ["TELEGRAM_TOKEN"]
CHANNEL = os.environ["CHANNEL_ID"]
TWELVE_KEY = os.environ.get("TWELVEDATA_KEY", "")

PAIRS = {
    "XAUUSD": "XAUUSD=X",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
}
# Smallest average 5 minute candle range worth trading. Below this, the spread eats the trade.
# These are starting values. Review them after your first weeks of results.
MIN_ATR = {"XAUUSD": 0.60, "EURUSD": 0.00025, "GBPUSD": 0.00030, "USDJPY": 0.030}

FILE = "signals.csv"
SUMMARY_FILE = "last_summary.txt"
COLUMNS = ["time", "pair", "strategy", "direction", "entry", "stop",
           "target", "status", "closed_time", "result_r"]

MAX_PER_DAY = 10      # total signals per day, across all pairs
MAX_PER_SCAN = 2      # most signals in one 5 minute scan
COOLDOWN_MIN = 30     # wait before a new signal on the same pair
EXPIRE_MIN = 180      # close a trade that is still open after 3 hours
CANDLE_MIN = 5
STALE_MIN = 10        # signals older than this are dropped, and this is the valid time
STOP_ATR = 1.0        # stop loss distance, in average candle ranges
TARGET_R = 1.5        # target is 1.5 times the stop distance
FIRST_HOUR = 5        # 05 UTC is 6am Nigerian time
LAST_HOUR = 15        # last signals at 4:55pm Nigerian time
SUMMARY_HOUR = 16     # daily summary at 5pm Nigerian time

TEST = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
SOURCES = {}
ERRORS = {}
DISCLAIMER = "Not financial advice. Risk no more than 1% of your account per trade."
HEADING = "⚡ LASER AI VIP SIGNAL ⚡"


def send(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHANNEL, "text": text}, timeout=30)
        if r.status_code != 200:
            print("Telegram error:", r.status_code, r.text[:200])
    except Exception as e:
        print("Telegram error:", type(e).__name__)


def load_log():
    if os.path.exists(FILE):
        return pd.read_csv(FILE, dtype=str).fillna("")
    return pd.DataFrame(columns=COLUMNS)


def read_marker():
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE) as f:
            return f.read().strip()
    return ""


def fmt(pair, x):
    if pair == "XAUUSD":
        return f"{x:.2f}"
    if "JPY" in pair:
        return f"{x:.3f}"
    return f"{x:.5f}"


def fetch_twelve(ticker):
    symbol = ticker.replace("=X", "")
    symbol = symbol[:3] + "/" + symbol[3:]
    r = requests.get(
        "https://api.twelvedata.com/time_series",
        params={"symbol": symbol, "interval": "5min", "outputsize": 400,
                "timezone": "UTC", "apikey": TWELVE_KEY},
        timeout=30,
    )
    time.sleep(8)  # stay inside the limit of 8 requests per minute
    data = r.json()
    if "values" not in data:
        msg = str(data.get("message", data))
        if TWELVE_KEY:
            msg = msg.replace(TWELVE_KEY, "***")
        ERRORS[ticker] = msg[:150]
        print("Twelve Data message:", ERRORS[ticker])
        return None
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    df = df[["open", "high", "low", "close"]].astype(float)
    df.columns = ["Open", "High", "Low", "Close"]
    return df


def fetch_yahoo(ticker):
    df = yf.download(ticker, period="5d", interval="5m",
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.dropna()
    idx = df.index
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return df


def get_data(pair, ticker):
    df = None
    source = "Twelve Data"
    if TWELVE_KEY:
        try:
            df = fetch_twelve(ticker)
        except Exception as e:
            ERRORS[ticker] = "connection problem (" + type(e).__name__ + ")"
            print("Twelve Data error:", type(e).__name__)
            df = None
    # Yahoo has no spot gold price, so gold has no backup feed
    if (df is None or df.empty) and pair != "XAUUSD":
        df = fetch_yahoo(ticker)
        source = "Yahoo"
    if df is None or df.empty:
        return None
    df = df.dropna()
    SOURCES[pair] = source

    # remove the candle that is still forming
    now = pd.Timestamp.now(tz="UTC")
    df = df[df.index + pd.Timedelta(minutes=CANDLE_MIN) <= now]
    if len(df) < 150:
        return None

    c, h, l = df["Close"], df["High"], df["Low"]
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()

    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()

    df["atr"] = atr
    df["atr_med"] = atr.rolling(100).median()
    df["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss)
    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["hh12"] = h.rolling(12).max().shift(1)
    df["ll12"] = l.rolling(12).min().shift(1)
    return df


def find_signal(pair, df):
    last = df.iloc[-1]
    need = ["atr", "atr_med", "adx", "rsi", "ema50", "hh12", "ll12"]
    if last[need].isna().any():
        return None
    # Quiet market, tiny candles or no trend: stay out
    if last["atr"] < 0.8 * last["atr_med"]:
        return None
    if last["atr"] < MIN_ATR[pair]:
        return None
    if last["adx"] < 20:
        return None

    up = last["ema9"] > last["ema21"] > last["ema50"]
    down = last["ema9"] < last["ema21"] < last["ema50"]
    recent = df.iloc[-3:]
    body = abs(last["Close"] - last["Open"])

    # Scalp 1: momentum pullback
    if up and recent["Low"].min() <= last["ema21"] and last["Close"] > last["ema9"] \
            and last["Close"] > last["Open"] and 45 <= last["rsi"] <= 68:
        return ("Momentum pullback", "BUY",
                "Short term uptrend, price dipped to the 21 average and bounced.")
    if down and recent["High"].max() >= last["ema21"] and last["Close"] < last["ema9"] \
            and last["Close"] < last["Open"] and 32 <= last["rsi"] <= 55:
        return ("Momentum pullback", "SELL",
                "Short term downtrend, price rose to the 21 average and turned down.")

    # Scalp 2: range breakout
    if up and last["adx"] >= 22 and last["Close"] > last["hh12"] and body >= 0.5 * last["atr"]:
        return ("Range breakout", "BUY",
                "Strong candle closed above the last hour's high in an uptrend.")
    if down and last["adx"] >= 22 and last["Close"] < last["ll12"] and body >= 0.5 * last["atr"]:
        return ("Range breakout", "SELL",
                "Strong candle closed below the last hour's low in a downtrend.")
    return None


def allowed(log, pair, strategy):
    # Learning step: pause a pair and strategy that lost over its last 15 trades
    done = log[(log["pair"] == pair) & (log["strategy"] == strategy)
               & (log["status"].isin(["win", "loss", "expired"]))]
    last15 = done.tail(15)
    if len(last15) >= 15 and pd.to_numeric(last15["result_r"]).sum() < 0:
        return False
    return True


def cooled_down(log, pair, now):
    rows = log[log["pair"] == pair]
    if rows.empty:
        return True
    last_time = pd.Timestamp(rows["time"].iloc[-1]).tz_localize("UTC")
    return now - last_time >= pd.Timedelta(minutes=COOLDOWN_MIN)


def close_at_market(log, i, row, df, now):
    entry = float(row["entry"])
    risk = abs(entry - float(row["stop"]))
    move = float(df["Close"].iloc[-1]) - entry
    if row["direction"] == "SELL":
        move = -move
    r = round(move / risk, 2)
    log.at[i, "status"] = "expired"
    log.at[i, "result_r"] = str(r)
    log.at[i, "closed_time"] = now.strftime("%Y-%m-%d %H:%M")
    return r


def main():
    now = pd.Timestamp.now(tz="UTC")
    hour, weekday = now.hour, now.weekday()
    today = now.strftime("%Y-%m-%d")
    log = load_log()
    last_summary = read_marker()
    messages = []

    data = {}
    for pair, ticker in PAIRS.items():
        try:
            data[pair] = get_data(pair, ticker)
        except Exception as e:
            print("Data error:", pair, type(e).__name__)
            data[pair] = None

    # 1. Check open trades
    for i, row in log[log["status"] == "open"].iterrows():
        df = data.get(row["pair"])
        if df is None:
            continue
        stop = float(row["stop"])
        target = float(row["target"])
        entry_time = pd.Timestamp(row["time"]).tz_localize("UTC")
        later = df[df.index >= entry_time]
        closed = False
        for ts, candle in later.iterrows():
            if row["direction"] == "BUY":
                hit_stop = candle["Low"] <= stop
                hit_target = candle["High"] >= target
            else:
                hit_stop = candle["High"] >= stop
                hit_target = candle["Low"] <= target
            # if both happen in one candle, count it as a loss to stay honest
            if hit_stop:
                log.at[i, "status"], log.at[i, "result_r"] = "loss", "-1"
                messages.append(f"❌ RESULT: {row['pair']} {row['direction']} hit the stop loss. Loss of 1R.")
                closed = True
            elif hit_target:
                log.at[i, "status"], log.at[i, "result_r"] = "win", str(TARGET_R)
                messages.append(f"✅ RESULT: {row['pair']} {row['direction']} hit the target. Gain of {TARGET_R}R 🎉")
                closed = True
            if closed:
                log.at[i, "closed_time"] = ts.strftime("%Y-%m-%d %H:%M")
                break
        if not closed and now - entry_time > pd.Timedelta(minutes=EXPIRE_MIN):
            r = close_at_market(log, i, row, df, now)
            messages.append(f"RESULT: {row['pair']} {row['direction']} closed after {EXPIRE_MIN} minutes at {r:+.1f}R.")

    # 2. Look for new signals
    in_window = weekday < 5 and FIRST_HOUR <= hour <= LAST_HOUR
    can_signal = TEST or in_window
    if os.path.exists("PAUSE") and not TEST:
        can_signal = False

    new_count = 0
    if can_signal:
        sent_today = int((log["time"].str[:10] == today).sum())
        candidates = []
        for pair, df in data.items():
            if df is None:
                continue
            age = now - (df.index[-1] + pd.Timedelta(minutes=CANDLE_MIN))
            if age > pd.Timedelta(minutes=STALE_MIN):
                continue  # data is old or the market is closed
            if ((log["pair"] == pair) & (log["status"] == "open")).any():
                continue
            if not cooled_down(log, pair, now):
                continue
            sig = find_signal(pair, df)
            if sig and allowed(log, pair, sig[0]):
                # no pair gets priority: the strongest trend goes first
                candidates.append((-float(df["adx"].iloc[-1]), pair, sig, df))
        candidates.sort(key=lambda x: x[0])

        for _, pair, (strategy, direction, reason), df in candidates:
            if new_count >= MAX_PER_SCAN or sent_today >= MAX_PER_DAY:
                break
            last = df.iloc[-1]
            entry = float(last["Close"])
            risk = STOP_ATR * float(last["atr"])
            sign = 1 if direction == "BUY" else -1
            stop = entry - sign * risk
            target = entry + sign * TARGET_R * risk
            entry_time = df.index[-1] + pd.Timedelta(minutes=CANDLE_MIN)
            log.loc[len(log)] = [entry_time.strftime("%Y-%m-%d %H:%M"), pair, strategy,
                                 direction, entry, stop, target, "open", "", ""]
            new_count += 1
            sent_today += 1

            go_time = (entry_time + pd.Timedelta(hours=1)).strftime("%H:%M")
            until = (entry_time + pd.Timedelta(minutes=STALE_MIN, hours=1)).strftime("%H:%M")
            if direction == "BUY":
                head = f"🟢 BUY {pair} 📈🔥"
            else:
                head = f"🔴 SELL {pair} 📉🔥"
            messages.append(
                f"{HEADING}\n\n"
                f"{head}\n\n"
                f"⏰ Enter at: {go_time} Nigerian time\n"
                f"⌛ Valid until: {until}\n\n"
                f"💰 Entry: around {fmt(pair, entry)}\n"
                f"🛑 Stop loss: {fmt(pair, stop)}\n"
                f"🎯 Target: {fmt(pair, target)}\n"
                f"⚖️ Reward to risk: {TARGET_R} to 1\n\n"
                f"🧠 Strategy: {strategy}\n"
                f"💡 Why: {reason}\n\n"
                f"😊 Skip it if price has moved more than {fmt(pair, 0.25 * risk)} "
                f"from the entry, or the valid time has passed.\n"
                f"⚠️ {DISCLAIMER}"
            )

    # Test run: show where prices came from and how fresh they are
    if TEST:
        lines = []
        for pair, df in data.items():
            if df is None:
                lines.append(f"{pair}: NO DATA. {ERRORS.get(PAIRS[pair], '')}")
            else:
                age = (now - (df.index[-1] + pd.Timedelta(minutes=CANDLE_MIN))).total_seconds() / 60
                lines.append(f"{pair}: {SOURCES.get(pair, '?')}, latest candle {age:.0f} min old")
        messages.append("TEST OK. The bot is connected.\n" + "\n".join(lines) +
                        "\n\nIf a candle is older than 10 minutes, the bot will not post signals on that pair.")

    # 3. Daily summary at 5pm Nigerian time
    if not TEST and weekday < 5 and hour >= SUMMARY_HOUR and last_summary != today:
        # close anything still open, so the day ends clean
        for i, row in log[log["status"] == "open"].iterrows():
            df = data.get(row["pair"])
            if df is not None:
                close_at_market(log, i, row, df, now)

        todays = log[log["time"].str[:10] == today]
        lines = []
        wins = losses = closed = 0
        net = 0.0
        for _, row in todays.iterrows():
            status = row["status"]
            r = float(row["result_r"]) if row["result_r"] != "" else 0.0
            if status == "win":
                tag, wins = "WON ✅", wins + 1
            elif status == "loss":
                tag, losses = "LOST ❌", losses + 1
            elif status == "expired":
                tag, closed = "CLOSED", closed + 1
            else:
                tag = "STILL OPEN"
            if status in ("win", "loss", "expired"):
                net += r
            extra = f" ({r:+.1f}R)" if status != "open" else ""
            local = (pd.Timestamp(row["time"]) + pd.Timedelta(hours=1)).strftime("%H:%M")
            lines.append(f"{local} {row['pair']} {row['direction']}: {tag}{extra}")

        text = f"📊 FOREX DAILY SUMMARY {today}\n"
        if len(todays) == 0:
            text += "No signals today. Staying out is part of the plan."
        else:
            text += (f"Signals: {len(todays)}\n"
                     f"Won: {wins}\nLost: {losses}\nClosed at day end: {closed}\n"
                     f"Net result: {net:+.1f}R\n\n"
                     "Times are Nigerian time.\n" + "\n".join(lines))
        if weekday == 4:
            done = log[log["status"].isin(["win", "loss", "expired"])]
            if len(done) > 0:
                total_r = pd.to_numeric(done["result_r"]).sum()
                w = int((done["status"] == "win").sum())
                text += (f"\n\nALL TIME\nClosed trades: {len(done)}\n"
                         f"Wins: {w} ({w / len(done) * 100:.0f}%)\n"
                         f"Total result: {total_r:+.1f}R")
        messages.append(text)
        last_summary = today

    for m in messages:
        send(m)

    log.to_csv(FILE, index=False)
    with open(SUMMARY_FILE, "w") as f:
        f.write(last_summary)


main()
