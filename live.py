import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import datetime
import joblib
import time
import argparse
import requests

# --- Configuration ---
SYMBOL = 'BTC-USD'
SEQ_LEN = 96
PRED_LEN = 48
FEATURES = 7
TARGET_IDX = 3 # Close price
SAVE_PATH = 'best_model.pkl'
SCALER_PATH = 'scaler.gz'

# Telegram Config (Use Environment Variables for security)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# --- Model Definition (Must match train.py) ---

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TemporalEmbedding, self).__init__()
        self.hour_embed = nn.Embedding(24, d_model)
        self.weekday_embed = nn.Embedding(7, d_model)
        self.day_embed = nn.Embedding(32, d_model)
        self.month_embed = nn.Embedding(13, d_model)

    def forward(self, x_mark):
        x_mark = x_mark.long()
        h = self.hour_embed(x_mark[:, :, 0])
        w = self.weekday_embed(x_mark[:, :, 1])
        d = self.day_embed(x_mark[:, :, 2])
        m = self.month_embed(x_mark[:, :, 3])
        return h + w + d + m

class CNN_GRU(nn.Module):
    def __init__(self, seq_len=96, pred_len=48, in_channels=7, d_model=128, conv_kernel_size=5, dropout=0.1):
        super(CNN_GRU, self).__init__()
        self.seq_len, self.pred_len, self.in_channels = seq_len, pred_len, in_channels
        self.temporal_embedding = TemporalEmbedding(d_model=d_model)
        self.value_embedding = nn.Linear(in_channels, d_model)
        
        # Stacked Conv1D layers (matching original depth pattern)
        padding = (conv_kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=conv_kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=conv_kernel_size, padding=padding)
        self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=conv_kernel_size, padding=padding)
        
        self.dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.fc = nn.Linear(d_model, in_channels)

    def pred_onestep(self, x_enc, x_mark):
        x = self.value_embedding(x_enc) + self.temporal_embedding(x_mark)
        x = self.dropout(x)
        
        # CNN Feature Extraction Stack
        x = x.permute(0, 2, 1) # [B, D, L]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.permute(0, 2, 1) # [B, L, D]
        
        # Sequential Memory
        out, _ = self.gru(x)
        return self.fc(out[:, -1:, :])

    def forward(self, x_enc, x_mark, y_mark):
        batch_size = x_enc.shape[0]
        pred_zero = torch.zeros(batch_size, self.pred_len, self.in_channels).to(x_enc.device)
        x_cat_pred = torch.cat([x_enc, pred_zero], dim=1)
        
        for i in range(self.pred_len):
            input_x = x_cat_pred[:, i:i + self.seq_len, :].clone()
            input_mark = y_mark[:, i:i + self.seq_len, :].clone()
            pred = self.pred_onestep(input_x, input_mark)
            x_cat_pred[:, self.seq_len + i:self.seq_len + i + 1, :] = pred
            
        return x_cat_pred[:, -self.pred_len:, :]

# --- Utilities ---

def send_telegram_msg(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

def add_technical_indicators(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df.fillna(method='bfill', inplace=True)
    return df

def get_time_features(dates):
    df_dates = pd.DataFrame(index=range(len(dates)))
    df_dates['date'] = pd.to_datetime(dates)
    df_dates['hour'] = df_dates['date'].dt.hour
    df_dates['weekday'] = df_dates['date'].dt.weekday
    df_dates['day'] = df_dates['date'].dt.day
    df_dates['month'] = df_dates['date'].dt.month
    return df_dates[['hour', 'weekday', 'day', 'month']].values

def get_live_data():
    print(f"Fetching latest {SYMBOL} data...", flush=True)
    df = yf.download(tickers=SYMBOL, interval='15m', period='60d', progress=False)
    
    if df.empty:
        raise ValueError("No data returned from yfinance.")

    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex): 
        df.columns = [col[0] for col in df.columns]
    
    date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df.rename(columns={date_col: 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df = add_technical_indicators(df)
    
    cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'EMA20']
    if not os.path.exists(SCALER_PATH): 
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
    
    scaler = joblib.load(SCALER_PATH)
    data = scaler.transform(df[cols].values)
    
    # Generate future stamps for recursive loop
    last_date = df['date'].iloc[-1]
    future_dates = [last_date + datetime.timedelta(minutes=15 * (i + 1)) for i in range(-SEQ_LEN, PRED_LEN)]
    full_stamps = get_time_features(future_dates)

    recent_data = torch.from_numpy(data[-SEQ_LEN:, :]).float().unsqueeze(0)
    full_stamp_tensor = torch.from_numpy(full_stamps).long().unsqueeze(0)
    
    return recent_data, full_stamp_tensor, scaler, df['Close'].iloc[-1].item(), last_date

def generate_live_signal():
    print(f"\n|{'=' * 50}|", flush=True)
    print(f"|{'Fetching Live Data & Generating Signal...':^50}|", flush=True)
    print(f"|{'=' * 50}|", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(SAVE_PATH):
        print(f"Error: Model not found at {SAVE_PATH}. Please run train.py first."); return

    try:
        data_tensor, stamp_tensor, scaler, current_price, last_time = get_live_data()
        model = CNN_GRU(SEQ_LEN, PRED_LEN, FEATURES).to(device)
        checkpoint = torch.load(SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint['model']); model.eval()

        with torch.no_grad():
            out = model(data_tensor.to(device), stamp_tensor[:, :SEQ_LEN, :].to(device), stamp_tensor.to(device))
            prediction = out[0, -1, TARGET_IDX].cpu().numpy()

        target_mean, target_scale = scaler.mean_[TARGET_IDX], scaler.scale_[TARGET_IDX]
        final_forecast = prediction * target_scale + target_mean
        diff = final_forecast - current_price
        
        signal = "BUY / UPTREND" if diff > 0 else "SELL / DOWNTREND"
        emoji = "🚀" if diff > 0 else "📉"
        
        report = (
            f"*BTC Trading Bot Signal*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🕒 *Time:* {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"💰 *Price:* ${current_price:,.2f}\n"
            f"🎯 *Forecast (12h):* ${final_forecast:,.2f}\n"
            f"📢 *SIGNAL:* `{signal}` {emoji}\n"
            f"📈 *Change:* `+{diff:.2f}`" if diff > 0 else f"📉 *Change:* `{diff:.2f}`\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )

        print(f"\n[LIVE SIGNAL REPORT] - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"Source Data End: {last_time}", flush=True)
        print(f"Current Price:  {current_price:.2f}", flush=True)
        print(f"Target Period:  12 Hours (48 x 15m bars)", flush=True)
        print(f"==> SIGNAL: {signal} ({diff:+.2f})", flush=True)
        print(f"==> Target Projection: {final_forecast:.2f}", flush=True)
        print(f"|{'=' * 50}|", flush=True)
        
        send_telegram_msg(report)
        
    except Exception as e:
        error_msg = f"Error during live signal generation: {e}"
        print(error_msg)
        # Optional: send_telegram_msg(f"⚠️ Bot Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loop', action='store_true', help='Run in a constant loop')
    parser.add_argument('--interval', type=int, default=900, help='Loop interval in seconds (default 900s / 15m)')
    args = parser.parse_args()

    if args.loop:
        print(f"Starting bot in LOOP mode (every {args.interval}s)...")
        while True:
            generate_live_signal()
            time.sleep(args.interval)
    else:
        generate_live_signal()
