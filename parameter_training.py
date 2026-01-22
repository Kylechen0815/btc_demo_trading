import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta, timezone
from colorama import Fore, Style, init
import optuna 
import time
import threading
from numba import njit

# 初始化顏色系統
init(autoreset=True)

# ==========================================
# --- ⚙️ 全域設定 ---
# ==========================================
MIN_TRADE = 60
START_DATE = "2025-12-15"
END_DATE = "2026-01-15"
STOP_REQUESTED = False

# ==========================================
# --- ⚡ Numba 極速運算引擎 (保持不動) ---
# ==========================================
@njit
def _numba_engine(data, sl_pct, tp_pct, move_pts, trade_qty, comm_pct, loss_pts, wide_t):
    pos = 0
    entry_p = 0.0
    entry_idx = -1
    trades = []
    for i in range(50, len(data) - 1):
        curr = data[i]; prev = data[i-1]; nxt = data[i+1]
        if pos != 0 and i > entry_idx:
            exited = False; pnl_pct = 0.0
            if pos == 1:
                tp_p = entry_p * (1 + tp_pct/100); sl_p = entry_p * (1 - sl_pct/100)
                if curr[1] >= tp_p: pnl_pct, exited = tp_pct, True
                elif curr[2] <= sl_p: pnl_pct, exited = -sl_pct, True
            else:
                tp_p = entry_p * (1 - tp_pct/100); sl_p = entry_p * (1 + sl_pct/100)
                if curr[2] <= tp_p: pnl_pct, exited = tp_pct, True
                elif curr[1] >= sl_p: pnl_pct, exited = -sl_pct, True
            if not exited:
                curr_pnl_pts = (curr[3] - entry_p) if pos == 1 else (entry_p - curr[3])
                if (abs(curr[3] - entry_p) >= move_pts and curr_pnl_pts > 0) or \
                   (abs(curr[3] - entry_p) >= loss_pts and curr_pnl_pts < 0):
                    if (pos == 1 and (prev[13] > 0 or prev[15] > 0)) or \
                       (pos == -1 and (prev[14] > 0 or prev[16] > 0)):
                        pnl_pct = ((nxt[0] - entry_p)/entry_p)*100 if pos == 1 else ((entry_p - nxt[0])/entry_p)*100
                        exited = True
            if exited:
                trades.append((trade_qty * (pnl_pct/100)) - (trade_qty * comm_pct * 2))
                pos, entry_idx = 0, -1
        if pos == 0:
            if curr[11] > 0 and curr[12] > 0 and (curr[4] - curr[5]) > wide_t:
                is_gs = (curr[3] > curr[0]) and (prev[3] > prev[0]); is_rs = (curr[3] < curr[0]) and (prev[3] < prev[0])
                is_bk = ((curr[3] > curr[4] and is_gs) or (curr[3] < curr[5] and is_rs))
                is_nr = (curr[8] >= 25 and curr[9] <= 0.002 and not is_bk)
                is_wd = (curr[8] >= 20 and curr[9] > 0.002 and not is_bk and curr[10] > 0)
                if (is_bk or is_nr or is_wd):
                    if curr[3] > curr[4] and curr[3] > curr[6]: pos, entry_p, entry_idx = 1, nxt[0], i + 1
                    elif curr[3] < curr[5] and curr[3] < curr[7]: pos, entry_p, entry_idx = -1, nxt[0], i + 1
    return trades

class BTCHeuristicOptimizer:
    def __init__(self):
        self.SYMBOL = 'BTC/USDT'; self.TIMEFRAME = '5m'; self.INIT_CAPITAL = 300.0
        self.TRADE_QTY_USD = 1500.0; self.COMMISSION_PCT = 0.0004
        self.exchange = ccxt.binance({'options': {'defaultType': 'future', 'adjustForTimeDifference': True}})
        self.LOOKBACK15 = 50; self.LOOKBACK5 = 10; self.CLIMAX_MULT = 4.5

    def fetch_data(self, start_str, end_str):
        tw_start = datetime.strptime(start_str, "%Y-%m-%d"); tw_end = datetime.strptime(end_str, "%Y-%m-%d")
        since = int((tw_start - timedelta(hours=8)).replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int((tw_end - timedelta(hours=8)).replace(tzinfo=timezone.utc).timestamp() * 1000)
        all_bars = []
        while since < end_ts:
            try:
                bars = self.exchange.fetch_ohlcv(self.SYMBOL, self.TIMEFRAME, since, 1000)
                if not bars: break
                since = bars[-1][0] + 1; all_bars += bars
            except: break
        return pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol'])

    def run_simulation(self, df_input, params):
        df = df_input.copy()
        df['body'] = (df['close'] - df['open']).abs(); df['avg_body'] = df['body'].rolling(20).mean()
        df['upper15'] = df['high'].shift(1).rolling(self.LOOKBACK15).max(); df['lower15'] = df['low'].shift(1).rolling(self.LOOKBACK15).min()
        df['m5_hh'] = df['high'].shift(1).rolling(self.LOOKBACK5).max(); df['m5_ll'] = df['low'].shift(1).rolling(self.LOOKBACK5).min()
        adx = ta.adx(df['high'], df['low'], df['close'], length=14); df['adx_v'] = adx['ADX_14']
        df['bbw_v'] = (df['high'].rolling(14).max() - df['low'].rolling(14).min()) / df['close'].rolling(14).mean()
        df['wide_vol_ok'] = (df['high'].rolling(5).max() - df['low'].rolling(5).min()) > 150
        er_l = int(params['er_l']); df['isEfficient'] = ((df['close'] - df['close'].shift(er_l)).abs() / (df['close'] - df['close'].shift(1)).abs().rolling(er_l).sum()) > params['er_t']
        r2_l = int(params['r2_l']); df['is_linear_trend'] = (df['close'].rolling(r2_l).corr(pd.Series(range(len(df)))) ** 2) > params['r2_t']
        h_l = df['high'] - df['low']; df['is_solid'] = np.where(h_l > 0, (df['body'] / h_l >= params['solid']), False)
        df['is_rev_r'] = df['is_solid'] & (df['close'] < df['open']) & (df['body'] > df['avg_body'] * self.CLIMAX_MULT)
        df['is_rev_g'] = df['is_solid'] & (df['close'] > df['open']) & (df['body'] > df['avg_body'] * self.CLIMAX_MULT)
        df['loss_l'] = (df['close'] < df['open']) & df['is_solid'] & (df['close'].shift(1) < df['open'].shift(1)) & df['is_solid'].shift(1)
        df['loss_s'] = (df['close'] > df['open']) & df['is_solid'] & (df['close'].shift(1) > df['open'].shift(1)) & df['is_solid'].shift(1)
        cols = ['open','high','low','close','upper15','lower15','m5_hh','m5_ll','adx_v','bbw_v','wide_vol_ok','isEfficient','is_linear_trend','is_rev_r','is_rev_g','loss_l','loss_s']
        return _numba_engine(df[cols].fillna(0).values.astype(np.float64), 
                     params['sl'], params['tp'], params['move'], 
                     self.TRADE_QTY_USD, self.COMMISSION_PCT, 
                     params['loss'], params['wide_t'])

def objective(trial, data):
    params = {
        'sl': 1.9, 'tp': 4.0, 'solid': 0.7, 'er_l': 10, 'er_t': 0.1, 'r2_l': 10, 'r2_t': 0.2,
        'move': trial.suggest_float('move', 500.0, 2000.0, step=100),
        'loss': trial.suggest_float('loss', 100.0, 500.0, step=50),
        'wide_t': trial.suggest_float('wide_t', 100.0, 600.0, step=50)
    }
    opt = BTCHeuristicOptimizer()
    trades = opt.run_simulation(data, params)
    
    equity = [300.0]
    for t in trades: equity.append(equity[-1] + t)
    profit = equity[-1] - 300.0
    
    trial.set_user_attr("n_trades", len(trades))
    trial.set_user_attr("profit_raw", profit)
    
    if len(trades) < MIN_TRADE: return -20000.0
    
    peak, mdd = 300.0, 0.1
    for e in equity: peak = max(peak, e); mdd = max(mdd, peak - e)
    r_sq = np.corrcoef(np.arange(len(equity)), equity)[0, 1] ** 2 if len(equity) > 5 else 0
    return (profit / mdd) * r_sq if profit > 0 else profit

def monitor_keyboard():
    global STOP_REQUESTED
    while True:
        if input().strip().lower() == 'y': STOP_REQUESTED = True; break

def stop_check_callback(study, trial):
    if STOP_REQUESTED: study.stop()

# ==========================================
# --- 🚀 執行主程序 (嚴格依照你提供的格式) ---
# ==========================================
if __name__ == "__main__":
    opt_tool = BTCHeuristicOptimizer()
    df_data = opt_tool.fetch_data(START_DATE, END_DATE)
    
    # 這裡依照你提供的參數設定
    sampler = optuna.samplers.TPESampler(n_startup_trials=200, multivariate=True, seed=78533)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    threading.Thread(target=monitor_keyboard, daemon=True).start()

    print(f"\n{Fore.GREEN}🚀 啟動優化任務 (隨時輸入 'Y' 停止)...")
    # 這裡依照你提供的 optimize 結構
    study.optimize(lambda trial: objective(trial, df_data), 
                   n_trials=2000, n_jobs=-1, callbacks=[stop_check_callback])

    # --- 最後顯示部分 (僅限於此進行排版修改) ---
    try:
        best = study.best_trial; p = best.params
        profit_display = best.user_attrs.get("profit_raw", 0.0)
        n_trades = best.user_attrs.get("n_trades", 0)

        print("\n" + "="*50)
        print(f"{Fore.YELLOW}🏆 優化參數結果:")
        print("-" * 50)
        print(f"min_move_pts  = {p['move']:.1f}")
        print(f"max_loss_pts  = {int(p['loss'])}")
        print(f"min_gap       = {int(p['wide_t'])}")
        print("-" * 50)
        
        # 顯示獲利金額，如果是負數則顯示紅色
        color = Fore.GREEN if profit_display >= 0 else Fore.RED
        print(f"{color}獲利金額: ${profit_display:.2f}")
        print(f"{Fore.WHITE}總交易筆數: {n_trades} 筆")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"尚未產生結果: {e}")
