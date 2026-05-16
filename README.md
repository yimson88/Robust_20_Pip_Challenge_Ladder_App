# Robust 20-Pip Challenge Ladder App

## Added from the image pattern

- Level-based challenge table
- One active trade only
- New signal validates only when no trade is active
- If TP is hit, the app moves to the next level
- If SL is hit, the app reverts to the previous level
- Lot size follows the current challenge level
- Current level can be set manually
- Challenge can be reset to Level 1

## State files

- data/challenge_state.csv
- data/active_signal_state.csv
- data/telegram_sent_signals.csv

## Telegram secrets for Streamlit Cloud

TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
