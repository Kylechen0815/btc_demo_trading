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

# ==========================================
# --- ⚙️ 全域設定 ---
# ==========================================
MIN_TRADE = 30
START_DATE = "2026-01-01"
END_DATE = "2026-01-15"
STOP_REQUESTED = False

# ==========================================
# --- ⚡ Numba 極速運算引擎 ---
# ==========================================
@njit
def _numba_engine(data, sl_pct, tp_pct, move_pts, trade_qty, comm_pct, loss_pts, wide_t):
    """
    data 矩陣索引: 0:open, 1:high, 2:low, 3:close, 4:upper15, 5:lower15, 
    6:m5_hh, 7:m5_ll, 8:adx_v, 9:bbw_v, 10:wide_vol_ok, 11:isEfficient, 
    12:is_linear_trend, 13:is_rev_r, 14:is_rev_g, 15:loss_l, 16:loss_s
    """
    pos = 0
    entry_p = 0.0
    entry_idx = -1
    trades = []
    
    for i in range(50, len(data) - 1):
        curr = data[i]
        prev = data[i-1]
        nxt = data[i+1]
        
        # 1. 出場判斷 (Exit Logic)
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
            
            # 移動利潤/虧損保護出場 (Morph Signal)
            if not exited:
                curr_profit_pts = (curr[3] - entry_p) if pos == 1 else (entry_p - curr[3])
                safe_exit = (abs(curr[3] - entry_p) >= move_pts) and (curr_profit_pts > 0)
                safe_loss_exit = (abs(curr[3] - entry_p) >= loss_pts) and (curr_profit_pts < 0)
                
                # 形態反轉判斷
                morph_sig = (pos == 1 and (prev[13] > 0 or prev[15] > 0)) or \
                            (pos == -1 and (prev[14] > 0 or prev[16] > 0))
                
                if (safe_exit or safe_loss_exit) and morph_sig:
                    pnl_pct = ((nxt[0] - entry_p)/entry_p)*100 if pos == 1 else ((entry_p - nxt[0])/entry_p)*100
                    exited = True
            
            if exited:
                net = (trade_qty * (pnl_pct/100)) - (trade_qty * comm_pct * 2)
                trades.append(net)
                pos, entry_idx = 0, -1

        # 2. 進場判斷 (Entry Logic)
        if pos == 0:
            # 核心修正：加入通道寬度門檻 $ChannelWidth = Upper15 - Lower15 > wide\_t$
            is_wide_channel = (curr[4] - curr[5]) > wide_t
            
            if curr[11] > 0 and curr[12] > 0 and is_wide_channel:
                is_gs = (curr[3] > curr[0]) and (prev[3] > prev[0])
                is_rs = (curr[3] < curr[0]) and (prev[3] < prev[0])
                
                is_bk = ((curr[3] > curr[4] and is_gs) or (curr[3] < curr[5] and is_rs))
                is_nr = (curr[8] >= 25 and curr[9] <= 0.002 and not is_bk)
                is_wd = (curr[8] >= 20 and curr[9] > 0.002 and not is_bk and curr[10] > 0)
                
                if (is_bk or is_nr or is_wd):
                    if curr[3] > curr[4] and curr[3] > curr[6]: # 做多
                        pos, entry_p, entry_idx = 1, nxt[0], i + 1
                    elif curr[3] < curr[5] and curr[3] < curr[7]: # 做空
                        pos, entry_p, entry_idx = -1, nxt[0], i + 1
                        
    return trades

# ==========================================
# --- 🏛️ 策略控制類 ---
# ==========================================
class BTCHeuristicOptimizer:
    def __init__(self):
        self.SYMBOL = 'BTC/USDT'
        self.TIMEFRAME = '5m'
        self.INIT_CAPITAL = 300.0   
        self.TRADE_QTY_USD = 1500.0 
        self.COMMISSION_PCT = 0.0004 
        # 修正：加入 adjustForTimeDifference 解決 -1021 錯誤
        self.exchange = ccxt.binance({
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
                'recvWindow': 10000
            }
        })
        self.LOOKBACK15 = 50
        self.LOOKBACK5 = 10
        self.CLIMAX_MULT = 4.5

    def fetch_data(self, start_str, end_str):
        tw_start = datetime.strptime(start_str, "%Y-%m-%d")
        tw_end = datetime.strptime(end_str, "%Y-%m-%d")
        utc_start = tw_start - timedelta(hours=8)
        utc_end = tw_end - timedelta(hours=8)
        since = int(utc_start.replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(utc_end.replace(tzinfo=timezone.utc).timestamp() * 1000)
        
        all_bars = []
        print(f"📡 正在抓取 {self.SYMBOL} 數據...")
        while since < end_ts:
            try:
                bars = self.exchange.fetch_ohlcv(self.SYMBOL, timeframe=self.TIMEFRAME, since=since, limit=1000)
                if not bars: break
                since = bars[-1][0] + 1
                all_bars += bars
                if len(bars) < 1000: break
            except Exception as e:
                print(f"抓取中斷: {e}")
                break
        df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'vol'])
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def run_simulation(self, df_input, params):
        df = df_input.copy()
        # 基本指標
        df['body'] = (df['close'] - df['open']).abs()
        df['avg_body'] = df['body'].rolling(20).mean()
        df['upper15'] = df['high'].shift(1).rolling(self.LOOKBACK15).max()
        df['lower15'] = df['low'].shift(1).rolling(self.LOOKBACK15).min()
        df['m5_hh'] = df['high'].shift(1).rolling(self.LOOKBACK5).max()
        df['m5_ll'] = df['low'].shift(1).rolling(self.LOOKBACK5).min()

        # 波動指標
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx_v'] = adx['ADX_14']
        df['bbw_v'] = (df['high'].rolling(14).max() - df['low'].rolling(14).min()) / df['close'].rolling(14).mean()
        df['wide_vol_ok'] = (df['high'].rolling(5).max() - df['low'].rolling(5).min()) > 150

        # 線性度與效率
        er_l = int(params['er_l'])
        net_chg = (df['close'] - df['close'].shift(er_l)).abs()
        total_chg = (df['close'] - df['close'].shift(1)).abs().rolling(er_l).sum()
        df['isEfficient'] = (net_chg / total_chg) > params['er_t']
        
        r2_l = int(params['r2_l'])
        df['bar_idx'] = range(len(df))
        df['r_squared'] = df['close'].rolling(r2_l).corr(df['bar_idx']) ** 2
        df['is_linear_trend'] = df['r_squared'] > params['r2_t']

        # 形態標籤
        h_l = df['high'] - df['low']
        df['is_solid'] = np.where(h_l > 0, (df['body'] / h_l >= params['solid']), False)
        df['is_rev_r'] = df['is_solid'] & (df['close'] < df['open']) & (df['body'] > df['avg_body'] * self.CLIMAX_MULT)
        df['is_rev_g'] = df['is_solid'] & (df['close'] > df['open']) & (df['body'] > df['avg_body'] * self.CLIMAX_MULT)
        df['loss_l'] = (df['close'] < df['open']) & df['is_solid'] & (df['close'].shift(1) < df['open'].shift(1)) & df['is_solid'].shift(1)
        df['loss_s'] = (df['close'] > df['open']) & df['is_solid'] & (df['close'].shift(1) > df['open'].shift(1)) & df['is_solid'].shift(1)

        cols = ['open','high','low','close','upper15','lower15','m5_hh','m5_ll',
                'adx_v','bbw_v','wide_vol_ok','isEfficient','is_linear_trend',
                'is_rev_r','is_rev_g','loss_l','loss_s']
        data_arr = df[cols].fillna(0).values.astype(np.float64)

        return _numba_engine(data_arr, params['sl'], params['tp'], params['move'], 
                             self.TRADE_QTY_USD, self.COMMISSION_PCT, params['loss'], params['wide_t'])

# ==========================================
# --- 🧪 Optuna 目標函數 ---
# ==========================================
def objective(trial, data):
    params = {
        'sl': trial.suggest_float('sl', 0.2, 3.0, step=0.1),
        'tp': trial.suggest_float('tp', 2.0, 6.0, step=0.1),
        'move': trial.suggest_float('move', 600.0, 1500.0, step=100),
        'solid': trial.suggest_float('solid', 0.4, 0.9, step=0.1),
        'er_l': trial.suggest_int('er_l', 10, 20),
        'er_t': trial.suggest_float('er_t', 0.1, 0.5, step=0.1),
        'r2_l': trial.suggest_int('r2_l', 5, 20),
        'r2_t': trial.suggest_float('r2_t', 0.1, 0.5, step=0.1),
        'loss': trial.suggest_float('loss', 100, 400.0, step=50),
        'wide_t': trial.suggest_float('wide_t', 100, 800, step=50) # 新增測試變數
    }
    
    opt = BTCHeuristicOptimizer()
    trades = opt.run_simulation(data, params)
    
    if len(trades) < MIN_TRADE: return -10000.0 + len(trades)
    
    equity = [300.0]
    for t in trades: equity.append(equity[-1] + t)
    
    profit = equity[-1] - 300.0
    peak, mdd = 300.0, 0.1
    for e in equity:
        peak = max(peak, e)
        mdd = max(mdd, peak - e)
    
    protection = 1.0 if min(equity) >= 200 else -5000.0
    calmar = profit / mdd if mdd > 0 else 0
    r_sq = np.corrcoef(np.arange(len(equity)), equity)[0, 1] ** 2 if len(equity) > 5 else 0
    
# --- 修正處：改為單數 set_user_attr 並分兩行寫 ---
    trial.set_user_attr("n_trades", len(trades))
    trial.set_user_attr("mdd", mdd)
    
    # 返回評分
    return calmar * r_sq * protection if profit > 0 else profit

# ==========================================
# --- ⌨️ 控制與監看系統 ---
# ==========================================
def monitor_keyboard():
    global STOP_REQUESTED
    while True:
        if input().strip().lower() == 'y':
            STOP_REQUESTED = True
            break

def stop_check_callback(study, trial):
    if STOP_REQUESTED: study.stop()

# ==========================================
# --- 🚀 執行主程序 ---
# ==========================================
if __name__ == "__main__":
    opt_tool = BTCHeuristicOptimizer()
    df_data = opt_tool.fetch_data(START_DATE, END_DATE)
    
    sampler = optuna.samplers.TPESampler(n_startup_trials=200, multivariate=True, seed=56465)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    threading.Thread(target=monitor_keyboard, daemon=True).start()

    print(f"\n{Fore.GREEN}🚀 啟動優化任務 (隨時輸入 'Y' 停止)...")
    study.optimize(lambda trial: objective(trial, df_data), 
                   n_trials=2000, n_jobs=-1, callbacks=[stop_check_callback])

    # 輸出結果
    try:
        best = study.best_trial
        p = best.params
        profit_final = best.user_attrs.get("profit", 0)
        
        print("\n" + "="*40)
        print(f"{Fore.YELLOW}🏆 最佳策略參數佈署格式:")
        print("-" * 40)
        # 按照你要求的格式輸出
        print(f"sl_pct        = {p['sl']:.1f}")
        print(f"tp_pct        = {p['tp']:.1f}")
        print(f"min_move_pts  = {p['move']:.1f}")
        print(f"solid_ratio   = {p['solid']:.2f}  // 同步 Python 邏輯")
        print(f"er_length     = {int(p['er_l'])}")
        print(f"er_thr        = {p['er_t']:.1f}")
        print(f"r2_len        = {int(p['r2_l'])}")
        print(f"r2_thr        = {p['r2_t']:.1f}")
        print(f"max_loss_pts  = {int(p['loss'])}")
        print(f"min_gap       = {int(p['wide_t'])}")
        print("-" * 40)
        print(f"{Fore.GREEN}💰 最終獲利總額: ${profit_final:.2f}")
        print(f"{Fore.CYAN}📊 總交易筆數: {best.user_attrs.get('n_trades')}")
        print("="*40)
    except Exception as e:
        print(f"尚未有最佳結果: {e}")
