import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta, timezone
from colorama import Fore, Style, init
import optuna 
import time
import sys
import os
import threading
from numba import njit, float64

# 初始化顏色系統
init(autoreset=True)
stop_requested = False
MIN_TRADE = 60
Start_date = "2025-12-15"
End_date = "2026-01-15"

# ==========================================
# --- ⚡ Numba 極速運算引擎 (核心邏輯) ---
# ==========================================
@njit
def _numba_engine(data, sl_pct, tp_pct, move_pts, trade_qty, comm_pct, loss_pts, wide_t, adx_t, bbw_t):
    """
    data 矩陣索引說明:
    0:open, 1:high, 2:low, 3:close, 4:upper15, 5:lower15, 6:m5_hh, 7:m5_ll, 
    8:adx_v, 9:bbw_v, 10:wide_vol_ok, 11:isEfficient, 12:is_linear_trend, 
    13:is_rev_r, 14:is_rev_g, 15:loss_l, 16:loss_s
    """
    pos = 0
    entry_p = 0.0
    entry_idx = -1
    trades = []

    for i in range(50, len(data) - 1):
        curr = data[i]
        prev = data[i-1]
        nxt = data[i+1]

        # 第一層邏輯：出場判斷
        if pos != 0 and i > entry_idx:
            exited = False
            pnl_pct = 0.0
            
            if pos == 1:
                tp_p = entry_p * (1 + tp_pct/100)
                sl_p = entry_p * (1 - sl_pct/100)
                if curr[1] >= tp_p: pnl_pct, exited = tp_pct, True
                elif curr[2] <= sl_p: pnl_pct, exited = -sl_pct, True
            else: # pos == -1
                tp_p = entry_p * (1 - tp_pct/100)
                sl_p = entry_p * (1 + sl_pct/100)
                if curr[2] <= tp_p: pnl_pct, exited = tp_pct, True
                elif curr[1] >= sl_p: pnl_pct, exited = -sl_pct, True

            if not exited:
                curr_profit_pts = (curr[3] - entry_p) if pos == 1 else (entry_p - curr[3])
                safe_exit = (abs(curr[3] - entry_p) >= move_pts) and (curr_profit_pts > 0)
                safe_loss_exit = (abs(curr[3] - entry_p) >= loss_pts) and (curr_profit_pts < 0)
                morph_sig = (pos == 1 and (prev[13] > 0 or prev[15] > 0)) or \
                            (pos == -1 and (prev[14] > 0 or prev[16] > 0))
                
                if (safe_exit and morph_sig) or (safe_loss_exit and morph_sig):
                    pnl_pct = ((nxt[0] - entry_p)/entry_p)*100 if pos == 1 else ((entry_p - nxt[0])/entry_p)*100
                    exited = True
                    
            if exited:
                net = (trade_qty * (pnl_pct/100)) - (trade_qty * comm_pct * 2)
                trades.append(net)
                pos, entry_idx = 0, -1

        # 第二層邏輯：進場判斷
        if pos == 0:
            is_wide_gap = (curr[4] - curr[5]) > wide_t
            
            if curr[11] > 0 and curr[12] > 0 and is_wide_gap:
                is_gs = (curr[3] > curr[0]) and (prev[3] > prev[0])
                is_rs = (curr[3] < curr[0]) and (prev[3] < prev[0])
                is_bk = ((curr[3] > curr[4] and is_gs) or (curr[3] < curr[5] and is_rs))
                # 這裡使用你指定的 ADX (25.15) 與 BBW (0.0039) 門檻
                is_nr = (curr[8] >= adx_t and curr[9] <= bbw_t and not is_bk)
                is_wd = (curr[8] >= 20 and curr[9] > bbw_t and not is_bk and curr[10] > 0)
                
                if (is_bk or is_nr or is_wd):
                    if curr[3] > curr[4] and curr[3] > curr[6]: 
                        pos, entry_p, entry_idx = 1, nxt[0], i + 1
                    elif curr[3] < curr[5] and curr[3] < curr[7]: 
                        pos, entry_p, entry_idx = -1, nxt[0], i + 1
                        
    return trades

class BTCHeuristicOptimizer:
    def __init__(self):
        self.SYMBOL = 'BTC/USDT'
        self.TIMEFRAME = '5m'
        self.INIT_CAPITAL = 300.0   
        self.TRADE_QTY_USD = 1500.0 
        self.COMMISSION_PCT = 0.0004 
        self.exchange = ccxt.binance({
            'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
        })
        # 指定固定常數
        self.LOOKBACK15 = 50
        self.LOOKBACK5 = 10
        self.CLIMAX_MULT = 4.5
        self.ADX_THRESHOLD = 25.15
        self.BBW_THRESHOLD = 0.0039

    def fetch_data(self, start_str, end_str):
        tw_start = datetime.strptime(start_str, "%Y-%m-%d")
        tw_end = datetime.strptime(end_str, "%Y-%m-%d")
        since = int((tw_start - timedelta(hours=8)).replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int((tw_end - timedelta(hours=8)).replace(tzinfo=timezone.utc).timestamp() * 1000)
        all_bars = []
        while since < end_ts:
            try:
                bars = self.exchange.fetch_ohlcv(self.SYMBOL, timeframe=self.TIMEFRAME, since=since, limit=1000)
                if not bars: break
                since = bars[-1][0] + 1; all_bars += bars
            except: break
        return pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol'])

    def run_simulation(self, df_input, params):
        df = df_input.copy()
        df['body'] = (df['close'] - df['open']).abs()
        df['avg_body'] = df['body'].rolling(20).mean()
        df['upper15'] = df['high'].shift(1).rolling(self.LOOKBACK15).max()
        df['lower15'] = df['low'].shift(1).rolling(self.LOOKBACK15).min()
        df['m5_hh'] = df['high'].shift(1).rolling(self.LOOKBACK5).max()
        df['m5_ll'] = df['low'].shift(1).rolling(self.LOOKBACK5).min()

        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx_v'] = adx['ADX_14']
        df['bbw_v'] = (df['high'].rolling(14).max() - df['low'].rolling(14).min()) / df['close'].rolling(14).mean()
        df['wide_vol_ok'] = (df['high'].rolling(5).max() - df['low'].rolling(5).min()) > 150

        # 使用指定常數固定指標計算
        er_l = int(params['er_l'])
        df['isEfficient'] = ((df['close'] - df['close'].shift(er_l)).abs() / (df['close'] - df['close'].shift(1)).abs().rolling(er_l).sum()) > params['er_t']
        r2_l = int(params['r2_l'])
        df['is_linear_trend'] = (df['close'].rolling(r2_l).corr(pd.Series(range(len(df)))) ** 2) > params['r2_t']

        h_l = df['high'] - df['low']
        df['is_solid'] = np.where(h_l > 0, (df['body'] / h_l >= params['solid']), False)
        df['is_rev_r'] = df['is_solid'] & (df['close'] < df['open']) & (df['body'] > df['avg_body'] * self.CLIMAX_MULT)
        df['is_rev_g'] = df['is_solid'] & (df['close'] > df['open']) & (df['body'] > df['avg_body'] * self.CLIMAX_MULT)
        df['loss_l'] = (df['close'] < df['open']) & df['is_solid'] & (df['close'].shift(1) < df['open'].shift(1)) & df['is_solid'].shift(1)
        df['loss_s'] = (df['close'] > df['open']) & df['is_solid'] & (df['close'].shift(1) > df['open'].shift(1)) & df['is_solid'].shift(1)

        cols = ['open','high','low','close','upper15','lower15','m5_hh','m5_ll','adx_v','bbw_v','wide_vol_ok','isEfficient','is_linear_trend','is_rev_r','is_rev_g','loss_l','loss_s']
        return _numba_engine(df[cols].fillna(0).values.astype(np.float64), 
                             params['sl'], params['tp'], params['move'], 
                             self.TRADE_QTY_USD, self.COMMISSION_PCT, params['loss'], params['wide_t'],
                             self.ADX_THRESHOLD, self.BBW_THRESHOLD)

# ==========================================
# --- 🧪 修改後：僅針對三項變數優化 ---
# ==========================================
def objective(trial, data):
    params = {
        # --- 固定參數清單 ---
        'sl': 1.9,
        'tp': 4.0,
        'solid': 0.7,
        'er_l': 18,
        'er_t': 0.1,
        'r2_l': 10,
        'r2_t': 0.2,
        
        # --- 🟢 僅優化這三個變數 ---
        'move': trial.suggest_float('move', 600.0, 1500.0, step=100),
        'loss': trial.suggest_float('loss', 100, 400.0, step=50),
        'wide_t': trial.suggest_float('wide_t', 50, 600, step=50)
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
    global stop_requested
    while True:
        if input().strip().lower() == 'y': stop_requested = True; break

def stop_check_callback(study, trial):
    if stop_requested: study.stop()

# ==========================================
# --- 🏁 執行區 (完全依照原始結構) ---
# ==========================================
if __name__ == "__main__":
    N_TRIALS = 100 
    main_opt = BTCHeuristicOptimizer()
    df_data = main_opt.fetch_data(Start_date, End_date)
    
    print(f"{Fore.CYAN}正在準備數據並鎖定常數執行優化...")
    
    sampler = optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True, seed=56465) 
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    threading.Thread(target=monitor_keyboard, daemon=True).start()

    print(f"\n{Fore.GREEN}🚀 啟動優化任務 (總計 {N_TRIALS} 次)...")
    study.optimize(lambda trial: objective(trial, df_data), 
                   n_trials=N_TRIALS, n_jobs=-1, callbacks=[stop_check_callback])

    # --- 🏁 顯示區域 ---
    try:
        best = study.best_trial; p = best.params
        profit_final = best.user_attrs.get("profit_raw", 0.0)
        
        print("\n" + "="*50)
        print(f"{Fore.YELLOW}🏆 優化結果 (對齊 Pine Script 格式):")
        print("-" * 50)
        # 顯示優化出來的變數
        print(f"min_move_pts  = {p['move']:.1f}")
        print(f"max_loss_pts  = {int(p['loss'])}")
        print(f"min_gap       = {int(p['wide_t'])}")
        print("-" * 50)
        
        color = Fore.GREEN if profit_final >= 0 else Fore.RED
        print(f"{color}獲利金額: ${profit_final:.2f}")
        print(f"{Fore.WHITE}總交易筆數: {best.user_attrs.get('n_trades')} 筆")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"尚未有最佳結果: {e}")
         


        11111